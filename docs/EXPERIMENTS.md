# Experiments (Core)

Date: April 28, 2026
Evidence organization last refreshed: `2026-04-28 12:06 EDT`
Paper-critical live queue status last refreshed: `2026-04-28 19:09 EDT`

## Current Status Summary

Problem we are solving:
- Reorganize the experiment record so the NeurIPS experiments section follows
  the paper's causal chain: sparse supports agree with basin labels when labels
  are used only for evaluation, those same supports route useful local
  predictors without oracle labels, and
  sparse-latent Koopman models remain competitive at long horizons.
- Make the motivation for each experiment clear to an external reader: support
  agreement with basin labels is a static membership test, while support-routed
  prediction is a separate dynamical test of whether the same support selects
  useful latent coordinates or local linear laws.

Current paper-facing approach:
- Use [PAPER_EXPERIMENT_EVIDENCE_MAP.md](/home/mila/l/lia/skae/docs/PAPER_EXPERIMENT_EVIDENCE_MAP.md)
  as the evidence-order hub for drafting.
- Keep this file as the detailed experiment log and keep
  [PAPER_TRACK_STATUS.md](/home/mila/l/lia/skae/docs/PAPER_TRACK_STATUS.md)
  as the high-level paper-track source of truth.
- Keep training/deployment claims label-free: basin labels and basin counts are
  used only for benchmark evaluation, not for training, model selection, or
  routing.

Solution/status:
- The live documentation is now organized around four evidence buckets:
  support agreement with basin labels, non-oracle support-routed local prediction,
  long-horizon forecasting competitiveness, and supporting/falsification
  evidence.
- [PAPER_EXPERIMENT_EVIDENCE_MAP.md](/home/mila/l/lia/skae/docs/PAPER_EXPERIMENT_EVIDENCE_MAP.md)
  now also contains the recommended main-text and appendix figure/table plan.
- The per-basin deep-slice interpretability re-evaluation is complete. Shard
  jobs `9388212`, `9388213`, `9388215`, `9388216`, and `9388217` and merge
  jobs `9388214` and `9388218` all completed with exit `0:0`; merged
  per-basin-slice CSVs now exist for the LISTA and boundary-MLP packets.
- The Table 2-4 statistical-testing audit is complete for the current
  manuscript tables. The completed `9388212-9388218` jobs are per-basin
  interpretability rows, not routing/refresh/Dysts seed packets, so they
  should be used for Table 1 appendix robustness rather than combined into
  Tables 2-4. The manuscript-side confirmatory tests now use within-system
  paired tests: seed-paired Wilcoxon/Holm for Table 2 routing and Table 4
  Dysts, and transfer-pair Wilcoxon/Holm for Table 3 support refresh.
- The manuscript Table 2 display has been temporarily switched from `H1000`
  routed/global ratios to `H100` routed/global ratios because the existing
  seed-`0`--`9` packet gives much stronger within-system Wilcoxon/Holm support
  at the shorter horizon. The caption contains a `\todo{}` to revisit the
  long-horizon `H1000` version after the seed expansion lands.
- The Table 2 routing statistics source has now been widened to the five-model
  paper-facing roster. Existing `200k` hard-init MLP-control seeds `0`--`9`
  are queued for self-routed routing evaluation as shards `9395314`--`9395319`
  with merge job `9395320`; dependent stats-refresh job `9395334` will rerun
  `tools/per_system_paired_tests.py` after the merge. The training backfill
  for complete `n_seed=15` coverage is staged as queue job `9395321`.
- A matched-dimension LISTA-SB fairness sensitivity is queued as job
  `9395415`. It will generate a `17` systems x `15` seeds x `200k` task table
  for `lista_dense_softblock_signsplit_p256_hardinit_basin_partition`, changing
  only the paper-facing LISTA-SB recipe's latent dimension from `64` to `256`,
  then queue forecasting collection and non-oracle self-routed evaluation.
- A Dysts LISTA soft-block `d_z=256` add-on for Table 4 is queued via
  [scripts/queue_dysts_seq10_lista_softblock_p256_seeds0to14.sh](/home/mila/l/lia/skae/scripts/queue_dysts_seq10_lista_softblock_p256_seeds0to14.sh).
  Launcher job `9396890` completed with exit `0:0`, built a `12` systems x
  `15` seeds task table (`180` training tasks), and submitted training array
  `9396894` plus dependent long-horizon eval launcher `9396895`.
  The row uses `sequence_length=10`, sparsity coefficient `0.006`, dense LISTA
  sign-split supports, and a 16-block soft-block transition penalty at
  `d_z=256`, followed by the same `H5000`--`H60000` `long60` Dysts evaluation
  chain after training.

Outstanding problem:
- Convert the evidence map into final paper prose, tables, and figures while
  keeping the claim calibrated: exact-support evidence is strongest deep in
  basins and for top-`8` routing; support families are more robust but less
  LISTA-specific; true local-geometry recovery remains a secondary mixed
  diagnostic.
- Keep the text from implying that a good basin label is automatically a good
  predictor selector. A support may identify where a state is while prediction
  depends on inactive coordinates, continuous latent coefficients, or
  cross-coordinate coupling in the learned transition.
- The immediate display-production priority is Figure 1 support maps, Table 1
  fixed-`17` alignment/forecasting, Table 2 non-oracle routing, Figure 2
  support refresh/routing, and Table 3 Dysts long-horizon forecasting. The
  new Dysts table is now explicitly an `H<=60000` extrapolation-beyond-training
  stress test: training uses 30K-step Dysts source trajectories with
  `sequence_length=10`, while held-out evaluation uses a separate 60K test
  cache.
- For Table 1, the existing global deep-slice numbers remain the main-text
  source of truth. The completed per-basin deep-slice packet is an appendix
  robustness check showing the same diagnostic under a slice where each basin
  contributes deep states.
- For Tables 2-4, the outstanding issue is power/coverage rather than test
  definition. Table 2 now displays `H100`, where LISTA-SB exact-support routing
  clears Holm on most systems, but the intended long-horizon `H1000` version
  still has only `1`--`10` valid seed pairs per system after route-availability
  filtering, so Wilcoxon/Holm pass counts are conservative.
  Existing `200k` hard-init MLP-control seeds `0`--`9` are being added to the
  Table 2 routing path before the final `n_seed=15` statistical refresh.
  The additional outstanding fairness question is whether the current
  LISTA-SB advantage partly reflects its smaller `d_z=64` latent space rather
  than architecture or sparsity; job `9395415` queues the matched `d_z=256`
  sensitivity needed to answer that directly.
  Table 4 remains limited by the current `n_seed=10` exact-test floor until
  the in-flight `n_seed=15` Dysts rerun lands; the new LISTA soft-block
  `d_z=256` add-on is a separate sensitivity row and is not yet part of the
  current manuscript table.

### 2026-04-28: Table 2 display switched to H100 pending seed expansion

1. Concrete results:
- Recomputed the existing `results/transition_rich_self_routed_forecasting_20260420/self_routed_forecasting_rows.csv`
  at `H100`, using the same per-system seed-paired Wilcoxon/Holm statistic as
  the audited `H1000` table.
- LISTA-SB exact-support routing is confirmatory at `H100`: all-slice
  `support_gated_k` / `support_local_centered` are `0.47 [12/17]` and
  `0.45 [11/17]`; deep-slice values are `0.38 [14/17]` and
  `0.36 [15/17]`.
- LISTA-BD remains weak for exact-support gated/local routing but is strong at
  the family level: `0.10 [11/17]` all and `0.032 [12/17]` deep.
- Dense MLP stays near the no-improvement line for exact support routes and
  fails at the family route (`> 220` IQM ratio).

2. Context:
- This does not change the evaluator or the merged CSV. It changes the
  manuscript-facing Table 2 horizon from `H1000` to `H100` so that the table
  better supports the non-oracle local-law claim with the current paired-seed
  coverage.

3. Interpretation:
- The `H100` table is the cleanest current confirmatory evidence that the
  model-produced support can select a useful local predictor without oracle
  basin labels. The `H1000` effect sizes remain large for LISTA-SB but are
  underpowered at the per-system Wilcoxon/Holm level because fewer valid
  seed-paired ratios survive the long-horizon route-availability and validity
  filters.

4. Project implications:
- Main-text Table 2 can now be read as a statistically supported short-horizon
  routing diagnostic. It should not be over-written as a long-horizon
  deployment guarantee until the seed expansion and added sparse-MLP routing
  controls land.

5. Next steps:
- Queue or repair the intended Table 2 seed-`10`--`14` expansion at the
  paper-facing scope: `200k` steps, five models
  (`LISTA-SB`, `LISTA-BD`, Dense MLP, Sparse MLP, Sparse MLP-BD), and then
  rerun self-routed forecasting/statistics at `H100`, `H500`, and `H1000`.
- The already completed job array `9392598` is not enough for this purpose: it
  used the three original models and `num_steps=20000`, not the required
  five-model `200k` Table 2 expansion.
- Status update at `2026-04-28 16:27 EDT`: the five-model `200k` expansion is
  now queued through
  [scripts/queue_transition_rich_table2_5model_seed_backfill.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_table2_5model_seed_backfill.sh).
  Queue job `9395321` will wait for the expanded user job count to drop below
  `550`, then submit a `%64` GPU array with `433` tasks: seeds `10`--`14` for
  all five roots plus the `8` known missing seed-`0`--`9` hard-init MLP-control
  rows.

### 2026-04-28: Dysts re-train at sequence_length=10, 15 seeds, 12-system subset

- Decision: re-run the Dysts long-horizon track at `sequence_length=10` (was
  `8`) for the narrowed `5`-recipe paper-facing Dysts set. Per-system seed
  count goes from `10` to `15`
  so that within-system paired Wilcoxon + Holm at `alpha/12 ≈ 4.2e-3` has
  exact-test floor headroom (with `n=15`, the one-sided Wilcoxon `p`-floor is
  `1/2^15 ≈ 3e-5`, well below Holm).
- System scope: `12` Dysts systems (drop `Duffing`, `SprottTorus`,
  `RikitakeDynamo` from the prior `15`-system list to save compute):
  `dysts:Chua, dysts:Dadras, dysts:DequanLi, dysts:Hadley,
  dysts:LorenzCoupled, dysts:LuChenCheng, dysts:MultiChua, dysts:QiChen,
  dysts:Sakarya, dysts:SanUmSrisuchinwong, dysts:ShimizuMorioka,
  dysts:WangSun`.
- Re-encode periods reduced to `{50, 75, 100, 200, 400, 600, 1000}` (was
  `{1, 5, 10, 20, 40, 60, 80, 100, 200, 300, 400, 500, 1000}`). This is
  the new default `EvaluationSettings.dysts_periodic_reencode_periods`.
- Long-horizon eval is extended to `H5000, H10000, H20000, H30000, H40000,
  H50000, H60000` in a single rollout to `H=60000` with periodic re-encoding;
  MSE is read off the same trajectory at each requested horizon. This is an
  extrapolation-beyond-training-horizon test because the training cache remains
  the `full` 30K-step source trajectory profile, while evaluation uses the new
  eval-only `long60` Dysts cache profile (`steps=60000`, `trajectories=200`,
  `warmup=2000`). New default `DEFAULT_HORIZONS` in
  `tools/evaluate_dysts_long_horizon_run.py`.
- Recipe set narrowed from `7` to `5` to save compute and consolidate
  hyperparameters. Drop `lista_blockdiag_ns200k_denseopt_sc3em3` and
  `generic_sparse_blockdiag_ns200k_sc3em3` (the low-sparsity block-diagonal
  variants). Apply the high-sparsity coefficient `sc=0.006` uniformly to
  the remaining sparsity-coefficient recipes:
  `generic_sparse_ns200k_best` (was `0.0025`),
  `generic_sparse_blockdiag_ns200k_sc6em3` (already `0.006`),
  `lista_blockdiag_ns200k_denseopt_sc6em3` (already `0.006`),
  `lista_dense_promoted_stage4` (was `0.003`).
  `generic_sparse_sc0_ns200k_best` (the dense-MLP zero-sparsity baseline)
  stays at `sc=0.0`. Final task count: `5 recipes × 12 systems × 15 seeds
  = 900 training tasks`.
- Training launcher: `scripts/_queue_dysts_seq10_seeds0to14.sh` (sbatch
  wrapper around `queue_paper_followup_recipes.sh`). Output dir
  `/network/scratch/l/lia/skae/paper_followup_recipes_200k_seq10_seeds0to14_20260428/`.
  Cluster `MaxSubmitJobs=1000` and the in-flight `255`-task multibasin
  array forced a chunked submission: training array submitted as two
  array jobs over the same TSV: chunk 1 covers tasks `0-699`
  (`ARRAY_OFFSET=0`, `9392814`) and chunk 2 covers tasks `700-899`
  (`ARRAY_OFFSET=700`, submitted by the replacement orchestrator as
  `9393590` at 2026-04-28 12:51 EDT once the user job count dropped below
  `800`).
- After training completes, the long-horizon eval will be queued via
  `scripts/queue_dysts_long_horizon_eval.sh` with `OUTPUT_TAG=
  dysts_long_horizon_h5k_to_h60k_seq10`, `DYSTS_CACHE_PROFILE=long60`, and
  `HORIZONS="5000 10000 20000 30000 40000 50000 60000"`. The original
  orchestrator job `9392878` was canceled before it could submit chunk 2 or
  the invalid 100K/full-cache eval, and replacement orchestrator `9393138` was
  submitted with the corrected 60K eval path.
- Implications for the paper: Table 4 changes from `15`-system `n=10`-seed at
  horizons `5K/10K/20K/30K` to `12`-system `n=15`-seed at horizons
  `5K..60K` (seven horizons), with `sequence_length=10` and the reduced
  re-encode-period grid. Interpret this as held-out long-rollout stability out
  to twice the 30K training source horizon, not as a claim that training saw
  60K windows.

### 2026-04-28: Tables 2-4 within-system statistics audit

1. Concrete results:
- Re-ran `tools/per_system_paired_tests.py` on an `unkillable` compute
  allocation after fixing the routing deep-slice mapping from `q1` to `q4`.
  The self-routed evaluator defines `q1` as the lowest basin-margin quartile
  (`boundary`) and `q4` as the highest basin-margin quartile (`deep`).
- Table 2 routing now reports within-system seed-paired Wilcoxon/Holm
  `[K/17]` counts in each cell. LISTA-SB has large IQM improvements but only
  `0/17`, `3/17`, and `3/17` Holm-cleared systems on the full slice, and
  `2/17`, `3/17`, and `3/17` on the deep slice because route availability
  leaves few valid seed pairs per system. LISTA-BD family-local routing clears
  `2/17` systems on both slices; Dense MLP clears `0/17` in all cells.
- Table 3 support refresh now uses the displayed post-entry
  `current_support_gated_periodic` rows: `356` LISTA-SB and `396` LISTA-BD
  transfer rollouts per cell. Per-system transfer-pair Wilcoxon/Holm clears
  `11/12` systems for LISTA-SB at periods `1` and `10`, and `10/12` systems
  for LISTA-BD at periods `1` and `10`.
- Table 4 Dysts keeps seed-paired Wilcoxon/Holm vs Dense MLP as the
  confirmatory statistic. With current `n_seed=10`, no cell exceeds `1/15`
  Holm-cleared systems; directional sign counts are descriptive only until
  the `n_seed=15` rerun lands.

2. Context:
- The jobs `9388212-9388218` produced
  `interpretability_per_basin_deep_pass1/interpretability_rows.csv` for the
  LISTA and boundary-MLP fixed-`17` packets. Those files contain
  support-size, entropy, family, operator, and wrong-support ablation metrics;
  they do not contain self-routed forecasting, periodic-refresh, or Dysts
  forecasting rows.

3. Interpretation:
- The appropriate confirmatory unit differs by table. Table 2 and Table 4
  should test paired seeds within each system. Table 3 should test paired
  transfer comparisons within each system because the current refresh packet
  is seed-`0` but contains many controlled transfer pairs.
- Cross-system sign tests can explain directionality but should not be the
  paper-facing confirmatory statistic for Tables 2-4.

4. Project implications:
- The manuscript table captions and prose have been changed to foreground
  within-system Wilcoxon/Holm statistics.
- The completed per-basin deep-slice jobs strengthen Table 1 appendix
  robustness, not Tables 2-4 sample size.

5. Next steps:
- Use the in-flight Dysts `n_seed=15`, `seq_len=10`, `12`-system rerun for
  the final Table 4 statistical claims once it completes.
- If Table 2 needs stronger confirmatory counts, run additional self-routed
  forecasting seeds or adjust route-availability coverage; do not substitute
  cross-system sign tests for seed-paired within-system inference.

## Paper Evidence Map

The experiments section should be written in this order:

1. **Do sparse supports align with basins when labels are used only for evaluation?**
   Lead with fixed-`17` agreement between supports and basin labels and make
   active supports, not pre-specified latent blocks, the target interpretability
   object.
2. **Do those same supports select useful local predictors without oracle basin labels?**
   Lead with exact top-`8` self-routed forecasting and the direct
   periodic-support-refresh ablation; use support families as robustness
   evidence. This is the dynamical test that follows naturally after the
   static label-agreement test.
3. **Do sparse-latent Koopman models remain competitive for long-horizon forecasting?**
   Present Dysts as the external long-horizon stress test, not as direct
   support-label evidence.
4. **Other useful things.**
   Keep centered local-law diagnostics, true-geometry checks, controlled
   transfer, phase portraits, tuning provenance, and superseded packets as
   supporting or appendix material unless they answer a specific objection.

Detailed result summaries and artifact links live in
[PAPER_EXPERIMENT_EVIDENCE_MAP.md](/home/mila/l/lia/skae/docs/PAPER_EXPERIMENT_EVIDENCE_MAP.md),
including the recommended figure/table sequence for the experiments section.

## Detailed Live Notes

Fixed-`17` LISTA root/result lookup:
- Use [FIXED17_LISTA_RESULTS_INDEX.md](/home/mila/l/lia/skae/docs/FIXED17_LISTA_RESULTS_INDEX.md) as the one-page lookup for the paper-facing fixed-`17` LISTA roots, their packets, and their headline saved results.
- Use [SUPPORT_OBJECT_GLOSSARY.md](/home/mila/l/lia/skae/docs/SUPPORT_OBJECT_GLOSSARY.md) for the paper-facing definitions of `absolute:0.001`, `relative:0.1`, `topk:8`, exact support, support family, and dominant group.

Problem we are solving:
- Show on the fixed `17`-system interpretability shortlist both that the models
  learn identifiable support structure that agrees with basin labels and that
  some induced sparsity is essential for good finite-dimensional Koopman
  representations when multiple basins or fixed points coexist. LISTA is one
  structured way to induce that
  sparsity; it is not the claim by itself.

Current paper-facing approach:
- Keep the live branch fixed to the `17` selected systems and no others:
  `multiwell_strong_transition`, `gated_local_linear`,
  `gated_transfer_linear`, `arrested_spiral`, `cal_asymmetric_3`,
  `cal_high_cross_3`, `cal_hexagon_6`, `cal_octagon_8`, `cal_pentagon_5`,
  `cal_square_4`, `checkerboard_potential`, `duffing_triple_well`,
  `snic_multi`, `transition_routes_4`, `var_depth_gradient_4`,
  `var_diamond_4`, and `var_l_shape_5`.
- Treat support agreement with basin labels as the primary interpretability
  objective. Within that branch, the main downstream functional check should be
  long-horizon forecasting at `H100`, `H500`, and `H1000`: if sparsity helps
  because the Koopman representation retains which basin the system currently
  occupies and that information is dynamically useful, then it should improve
  long-horizon rollouts, not just short-horizon or one-step metrics.
- Treat this as two linked but non-identical objectives. First, the support
  should identify basin membership or a coherent part of a basin. Second, the
  support should be useful for selecting prediction behavior. The second does
  not follow automatically from the first, because the columns or subspaces of
  the learned transition that carry local dynamics need not be the same
  coordinates that mark basin membership.
- The local-law evidence currently answers a narrow version of the
  local-versus-global question. In the completed centered local-law packet,
  support-conditioned laws beat the model's learned global transition
  especially on states far from basin boundaries, but the comparison against a
  separately refit global-centered slope is weaker. Therefore the safe current
  statement is "support-conditioned laws beat the learned global Koopman law on
  the covered states," not "support-conditioned laws beat every possible global
  refit under equal regularization and data accounting."
- The two reviewer-response mechanism branches now have second-audited
  seed-`0` fixed-`17` outputs. The first outputs under
  `*_20260423_cached` and `*_20260423` are superseded for paper claims, and
  the first corrected `*_20260423_corrected` true-geometry packet is also
  superseded because it evaluated shared support/family objects only at their
  dominant fixed point.
  The protocols are
  [true_jacobian_geometry_experiment_20260423.md](/home/mila/l/lia/skae/docs/planning/true_jacobian_geometry_experiment_20260423.md)
  and
  [controlled_transfer_switching_experiment_20260423.md](/home/mila/l/lia/skae/docs/planning/controlled_transfer_switching_experiment_20260423.md).
  The current true-geometry packet is
  [results/true_jacobian_geometry_fixed17_seed0_20260424_reaudit](/home/mila/l/lia/skae/results/true_jacobian_geometry_fixed17_seed0_20260424_reaudit)
  with `49/49` runs complete, `198,302` rows, `114,419` ok rows, and `0`
  failures. The current controlled-transfer packet is
  [results/controlled_transfer_switching_fixed17_seed0_20260424_reaudit](/home/mila/l/lia/skae/results/controlled_transfer_switching_fixed17_seed0_20260424_reaudit)
  with all three root shards complete, `1,776` rows, `1,632` ok rows, `144`
  skipped rows, and `0` failures.
  April 25 verification: `sacct` shows jobs `9347926-9347929` completed with
  exit `0:0`, there are no matching jobs currently in `squeue`, and the
  controlled-transfer packet is stored as three completed root shards rather
  than a merged top-level summary file.
  Claim-framing note: the MLP roots are useful specificity controls, but they
  are not required for the narrower reviewer-response questions. For true
  geometry, the clean LISTA-only read is observed support/family partitions
  versus count-matched random partitions and attractor/basin baselines. For
  controlled transfer, the current packet tests whether the encoder's support
  objects switch on a measured state-space basin transfer; it is adjacent to,
  but not by itself a full periodic-reencoding rollout ablation.
- Direct periodic-support-refresh test: because the controlled-transfer packet
  does not by itself prove the stronger rollout mechanism, a dedicated
  post-entry ablation is now scaffolded and smoke-tested. The evaluator
  [evaluate_transition_rich_periodic_support_refresh.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_periodic_support_refresh.py)
  starts continuations at measured target entry and post-entry states, compares
  stale source-latent no-refresh, current-state no-refresh, periodic
  decode/re-encode, previous source-support gated `K`, and refreshed-support gated `K`,
  and records support-refresh events, route-target
  fraction, fallback/chatter, and post-entry forecast MSE. Smoke job `9361455`
  on `cal_square_4` completed with `32/32` ok rows and `0` failures under
  [results/periodic_support_refresh_smoke_20260425_cal_square](/home/mila/l/lia/skae/results/periodic_support_refresh_smoke_20260425_cal_square).
  The fixed-`17` seed-`0` LISTA-only packet completed its two science shards
  cleanly under
  [results/periodic_support_refresh_fixed17_seed0_20260425](/home/mila/l/lia/skae/results/periodic_support_refresh_fixed17_seed0_20260425).
  Dense LISTA shard `9361464` completed `16/16` specs with `34,440` rows
  (`34,176` ok, `264` skipped, `0` failures), and blockdiag LISTA shard
  `9361465` completed `17/17` specs with `38,280` rows (`38,016` ok,
  `264` skipped, `0` failures). Merge job `9361470` is pending on priority,
  but the per-root summaries are already available.
  Main result: the key claim is positively addressed for the known strongest
  dense LISTA exact-support setup, especially `topk:8` after the trajectory is
  clearly in the target basin. Dense LISTA exact `topk:8` post-start
  refreshed-support gating reaches route-target fraction
  `0.8552/0.8886`, fallback `0.1392/0.1058`, and refreshed-versus-previous-support
  MSE ratio `0.0093/0.0131`. Dense LISTA `topk:8` family is even cleaner:
  route-target fraction `1.0000/0.9996`, and refreshed-versus-previous-support MSE ratio
  `0.000525/0.000626`. The result is not uniformly positive for every LISTA
  setup: blockdiag exact supports remain weak as target objects after
  re-encoding, although blockdiag support families are strongly positive.
- Second-audited true-geometry result: the strongest safe read is still
  local-chart evidence, not true-Jacobian recovery. LISTA family partitions
  often beat random count-matched partitions near attractors, but the margin is
  weaker after class-attractor accounting. At radius `0.15`, blockdiag LISTA
  family relative-Frobenius error is `0.1169` vs `0.1979` random for
  `absolute:0.001`, `0.1205` vs `0.1872` for `relative:0.1`, and `0.1114` vs
  `0.1955` for `topk:8`. Dense LISTA `topk:8` family is
  `0.1328 / 0.1268 / 0.1304` vs random `0.1541 / 0.1404 / 0.1394`, and dense
  LISTA exact `topk:8` support is `0.1410 / 0.1329 / 0.1512` vs random
  `0.1881 / 0.1591 / 0.1558`. The zero-sparse MLP remains a warning: its
  `topk:8` family rows also beat random at radii `0.15` and `0.3`, while its
  exact `topk:8` support rows remain worse than random. Therefore this branch
  should be written as a secondary falsification diagnostic, not a headline
  result or a broad LISTA-specific true-Jacobian claim.
- Second-audited controlled-transfer result: the paper-positive exact-support read
  is specific to dense LISTA under `topk:8`. Dense LISTA `topk:8` exact
  supports have transfer pre-source dominance `0.8194`, post-target dominance
  `0.8230`, post-bridge target dominance `0.9370`, post-bridge lag `6.0455`
  steps, and chatter `0.0375`. The matched zero-sparse MLP `topk:8` exact
  support is weak (`0.3710`, `0.3114`, `0.3504`, lag `9.0455`), and
  blockdiag LISTA exact support collapses (`0.0172`, `0.0448`, `0.0519`).
  Support-family switching is clean for all three roots under `topk:8`
  post-bridge target dominance (`1.0000`, `0.9989`, `1.0000` for dense LISTA,
  blockdiag LISTA, and zero-sparse MLP), so family-level transfer switching is
  useful evidence that learned representations track basin changes, but it is
  not by itself LISTA-specific induced-sparsity evidence.
- Treat the causal comparison as “induced sparsity versus no induced sparsity,”
  not just “LISTA versus MLP.” The matched sparse MLP control already tests
  whether LISTA's encoder structure matters beyond sparsity itself; the clean
  tanh / no-shrink MLP control is the decisive test of whether removing
  induced sparsity degrades forecasting and basin-local support stability.
- For paper-facing comparisons that are meant to isolate architecture, keep the
  training sampling regime identical across compared roots. Treat
  near-separatrix hard-init oversampling as its own experiment axis, report it
  as a separate table entry or column, and do not read hard-init LISTA versus
  standard-sampling MLP as an architecture-only contrast.
- Keep the older locked-budget `generic_sparse` plus `sparsity_coeff=0.0`
  packet as scientifically meaningful supporting evidence even though it is
  not the clean tanh / no-shrink control. It isolates the effect of removing
  the explicit `L1` penalty while retaining ReLU-induced architectural
  shrinkage, so it is still informative about whether that weaker form of
  induced sparsity changes basin-support identification or forecasting.
- The supporting benchmark visual packet for senior coauthors is now complete
  at
  [H5000](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h5000_lista_phase_portraits_manifest.json),
  [H20000](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h20000_lista_phase_portraits_manifest.json),
  the older LISTA-only shared-batch
  [H30000](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h30000_lista_phase_portraits_manifest.json),
  and the benchmark-aligned all-roots
  [H30000 best-root packet](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h30000_best_root_phase_portraits_manifest.json).
  The new all-roots H30000 selector uses the completed seven-root collector
  rows and picks the lowest `H30000` best-periodic MSE per system across all
  checked-in roots, seeds, and periodic cadences; it selects block-diagonal
  LISTA on `14/15` systems (`sc=6e-3` on `10`, `sc=3e-3` on `4`) and dense
  LISTA only on `dysts:QiChen`. Use that packet as the preferred `H30000`
  visual appendix. Keep the older dense-heavy `H30000` shared-batch LISTA
  packet only as a selection-rule sensitivity artifact.
- The first full seed-`10` Dysts long-horizon reevaluation packet under
  [results/dysts_long_horizon_eval_20260414](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414)
  is now complete. The refreshed collector summary
  [summary.json](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/collect/summary.json)
  reports `750/750` complete tasks and `0/750` pending or invalid tasks
  across the five verified benchmark roots.
- The cache-length fix in
  [skae/data.py](/home/mila/l/lia/skae/skae/data.py) did hold in production:
  cache prebuild `9273655`, validation `9273656`, collector `9273658`, and
  the replacement validation probe `9273675` all completed cleanly. The only
  broad failure was the main reevaluation array `9273657`, so the remaining
  issue at that point was rescue coverage rather than broken `H30000`
  evaluation plumbing.
- The main rescue root cause is now identified and patched in
  [skae/config.py](/home/mila/l/lia/skae/skae/config.py): many older
  checkpoints serialize legacy environment keys such as
  `ENV.COMPETITIVE_LV.SYSTEM_SEED`, and the old `Config.from_dict()` path
  raised on those unknown fields before evaluation started. The loader now
  filters unknown dataclass kwargs for backward-compatible checkpoint reads.
- That compatibility fix is already validated on a previously failing task:
  zero-sparse MLP `dysts:Chua` seed `0` now reevaluates cleanly and writes
  [evaluation_results_checkpoint.json](/network/scratch/l/lia/skae/paper_zero_sparse_benchmark_200k_20260321/paper_zero_sparse_benchmark/generic_sparse_sc0_ns200k_best/dysts_Chua/dt_0p0002847474579095888/seed_0/20260321-152929/reeval_dysts_long_horizon_h5000_h10000_h20000_h30000/evaluation_results_checkpoint.json).
- Rescue pass `1` is now complete. Reevaluation array `9278881` and recollect
  `9278882` both finished cleanly, and the refreshed collector under
  [results/dysts_long_horizon_eval_20260414/collect](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/collect)
  now reports `750/750` complete tasks and `0` pending or invalid tasks. The
  first full seed-`10` Dysts long-horizon packet is therefore complete rather
  than partial.
- On the full `750/750` collector, dense LISTA is the best aggregate root at
  `H5000/H10000` with median best-periodic MSE `0.1285/0.9778`, while
  block-diagonal LISTA `sc=6e-3` is the best aggregate root at
  `H20000/H30000` with `1.9150/2.2720`. The two MLP controls are now fully
  populated too: sparse MLP gives `0.1953/1.2373/3.2524/3.6981`, and
  zero-sparse MLP gives `0.2474/1.4564/3.2354/3.7893`. The supporting Dysts
  benchmark is therefore now complete enough for paper-facing interpretation,
  even though the fixed-`17` causal branch remains the lead storyline.
- The paper Dysts launcher itself now supports the missing block-diagonal MLP
  family directly. `generic_sparse_blockdiag` is now a first-class paper
  benchmark variant in
  [skae/benchmarks/paper_benchmark_manifest.py](/home/mila/l/lia/skae/skae/benchmarks/paper_benchmark_manifest.py),
  and
  [scripts/queue_paper_followup_recipes.sh](/home/mila/l/lia/skae/scripts/queue_paper_followup_recipes.sh)
  now keeps the canonical comparison anchors stable even on partial reruns,
  exposes `ARRAY_PARALLEL`, and falls back to benchmark-default `dt` values
  when the historical selected-`dt` table is absent instead of hard-failing.
- The Dysts block-diagonal MLP short-horizon packet under
  [results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415)
  is now collected. Replacement wrapper `9282328`, training array `9282330`,
  collector `9282331`, and compare jobs `9282332-9282334` all finished.
  The collected Dysts-only packet covers `H100/H500/H1000` in
  [paper_benchmark_summary.md](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415/collect/paper_benchmark_summary.md)
  and
  [forecasting_rows.csv](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415/collect/forecasting_rows.csv).
  Dysts-only system-median best-periodic MSE is
  `9.227e-05 / 0.001412 / 0.004684` for `generic_sparse_blockdiag_ns200k_sc3em3`
  and `7.454e-05 / 0.001399 / 0.004821` for
  `generic_sparse_blockdiag_ns200k_sc6em3`, so `sc=6e-3` is slightly better
  in aggregate at `H100/H500` while `sc=3e-3` is slightly better at `H1000`.
- The Dysts blockdiag-MLP long-horizon extension is now complete under
  [results/dysts_long_horizon_eval_mlp_blockdiag_20260415](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_mlp_blockdiag_20260415).
  Retry `9286093_150` completed in `43m49s`, launcher `9286094` completed, and
  the chained cache/validate/eval/collect jobs `9289755-9289758` all finished
  cleanly. The collector
  [summary.md](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_mlp_blockdiag_20260415/collect/summary.md)
  reports `300/300` complete tasks and `0` pending or invalid tasks across the
  two new roots.
- The long-horizon reevaluation limit is now reduced to `03:00:00` in the
  existing launcher path via
  [scripts/queue_dysts_long_horizon_eval.sh](/home/mila/l/lia/skae/scripts/queue_dysts_long_horizon_eval.sh).
  That change is empirically safe: across the completed `750/750` Dysts
  long-horizon packet, the completed `H30000` reevaluation tasks had
  `max=1825s` (`30m25s`), `p95=384s`, and `0/750` runs over `3` hours.
- On the full seven-root Dysts long-horizon comparison, dense LISTA remains
  the best aggregate root at `H5000/H10000` with `0.1285/0.9778`, while
  block-diagonal LISTA `sc=6e-3` remains the best aggregate root at
  `H20000/H30000` with `1.9150/2.2720`. The new blockdiag-MLP roots land at
  `0.1501/1.1401/3.0536/3.5891` (`sc=3e-3`) and
  `0.1945/1.2761/2.9519/3.4785` (`sc=6e-3`). So the blockdiag-MLP family is
  genuinely competitive at `H5000/H10000` but does not change the headline
  long-horizon winner: it is stronger than the plain MLP controls, yet still
  loses the `H20000/H30000` aggregate read to block-diagonal LISTA `sc=6e-3`.
- The reframed centered-chart mechanism packet is now complete under
  [results/transition_rich_centered_chart_mechanism_20260420](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420).
  Shards `9310546-9310548` and merge `9310549` all finished cleanly, and the
  merged packet writes
  [centered_chart_mechanism_rows.csv](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420/centered_chart_mechanism_rows.csv),
  [centered_chart_mechanism_summary.md](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420/centered_chart_mechanism_summary.md),
  and
  [manifest.json](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420/manifest.json)
  with `74,369` rows and `0` failures. This is the decision-grade rerun of the
  local-law question with centered local charts, explicit depth strata, and
  the actual dense `tanh` / no-shrink MLP control. On
  `relative:0.1` exact support, `persistent_current`, and deep `q4` states,
  centered support-conditioned local slopes beat the learned global `K` on
  `93.1%` of evaluated blockdiag LISTA seed-system rows (`130` rows),
  `98.6%` of dense LISTA rows (`141`), and `100%` of dense no-sparsity MLP
  rows (`140`). The direct support-gated `K` read is also strongly positive
  deep in-basin: q4 input-gated/global-`K` win rate is `100%` for blockdiag
  LISTA (`121` rows; mean ratio `0.010`), `97.8%` for dense LISTA (`139`;
  mean `0.210`, median `0.001`), and `100%` for the dense no-sparsity MLP
  (`140`; mean `0.014`).
- The next paper-critical read is no longer merely queued; it is complete
  under
  [results/transition_rich_self_routed_forecasting_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420).
  Shards `9314443-9314472` and merge `9314473` all finished cleanly, and the
  merged packet writes
  [self_routed_forecasting_rows.csv](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420/self_routed_forecasting_rows.csv),
  [self_routed_forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420/self_routed_forecasting_summary.md),
  and
  [manifest.json](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420/manifest.json)
  with `510/510` runs complete, `24,600` rows, and `0` failures. This is the
  direct deployment-facing test of whether the model's own support or support
  family can route forecasting better than one global `K` without oracle basin
  labels.

  The strongest paper-facing result is exact-support `topk:8` routing on the
  dense LISTA root. On `H1000/global`, all-slice median ratios / win rates are
  `0.228 / 0.920` for `support_gated_k` and `0.275 / 0.947` for
  `support_local_centered`, with median coverage about `0.53`. Deep `q4`
  states are slightly stronger at `0.224 / 0.923` and `0.207 / 0.985`. The
  matched zero-sparsity `tanh` MLP is much weaker on the same non-oracle
  router: all-slice `support_gated_k` is `0.924 / 0.539`, while
  `support_local_centered` is `1.000 / 0.496`; on deep `q4` it remains only
  `0.964 / 0.519` and `1.000 / 0.473`. So the dense LISTA encoder now has
  direct evidence that its own inferred support can select a useful local law
  for long-horizon forecasting without basin labels, and that the dense
  no-sparsity control does not provide a routing signal of comparable quality.

  The blockdiag LISTA root is also positive but weaker and lower-coverage:
  all-slice `H1000/global` medians / win rates are `0.832 / 0.739` for
  `support_gated_k`, `0.801 / 0.783` for `support_local_centered`, and
  `0.983 / 0.696` for the direct `support_block_gated_k` read, but median
  coverage is only about `0.12`, `0.12`, and `0.003`, respectively. This is
  still useful paper evidence because it keeps the direct support-gated-`K`
  mechanism alive, but it is not the strongest forecasting win.

  The exact-support `relative:0.1` router is not deployment-usable. On deep
  `q4`, all exact-support routed rows are skipped (`160/160` per root for
  `support_gated_k`) with `support_class_count>max_partition_classes`, so the
  thresholded exact support remains too fragmented to be the paper's main
  routing object. Support family is higher coverage and often strongly
  positive on LISTA medians (`H1000/global` all-slice medians
  `2.9e-4` blockdiag and `2.2e-3` dense; deep `q4` medians `2.6e-7` and
  `5.9e-6`), but the mean tables explode because a minority of catastrophic
  rollouts dominate. That heavy-tail instability is especially clear in the
  zero-sparsity MLP family router, which is outright bad (`H1000/global`
  median `54.6` all-slice and `47.6` on deep `q4`).

  The self-routed evaluator itself was also upgraded mid-flight for resumable
  intra-shard reruns with atomic per-spec flushing in
  [tools/evaluate_transition_rich_self_routed_forecasting.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_self_routed_forecasting.py)
  with shard / merge launchers
  [scripts/run_transition_rich_self_routed_forecasting.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_self_routed_forecasting.sh),
  [scripts/merge_transition_rich_self_routed_forecasting_shards.sh](/home/mila/l/lia/skae/scripts/merge_transition_rich_self_routed_forecasting_shards.sh),
  and
  [scripts/queue_transition_rich_self_routed_forecasting_shards.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_self_routed_forecasting_shards.sh).
  Smoke validation is complete under
  [results/transition_rich_self_routed_forecasting_smoke_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_smoke_20260420)
  with `270` rows and `0` failures across the reduced `3`-system / `3`-root /
  seed-`0` packet, and merge-path validation is complete under
  [results/transition_rich_self_routed_forecasting_merge_smoke_20260420/merged](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_merge_smoke_20260420/merged).
  Compute-node validation job `9315112` completed in `16s` on `cn-m004` and
  confirmed that rerunning the one-spec smoke shard simply resumes with `1/1`
  completed and `0` remaining.
- Outstanding problem:
  the non-oracle forecasting question is now answered, and the remaining
  blocker is claim calibration rather than missing evidence. The paper can now
  say three concrete things. First, support-selected local laws are usable
  without basin labels, most clearly for the dense LISTA `topk:8` router.
  Second, induced sparsity still matters for routing quality: the zero-sparse
  `tanh` MLP is materially weaker on the same self-routed exact-support read.
  Third, not every routing object is equally paper-worthy: thresholded exact
  support is too fragmented (`relative:0.1` skips), and support family has a
  catastrophic tail even when its medians are strong. The remaining work is to
  position `topk:8` exact support as the primary deployment router, keep
  family-level routing as a higher-coverage but unstable supporting result,
  and decide how prominently to retain the low-coverage blockdiag
  `support_block_gated_k` mechanism read.
- The fixed-`17` LISTA phase-portrait packet for senior-coauthor handoff is
  now also complete at
  [H1000/H3000/H5000](/home/mila/l/lia/skae/docs/figures/fixed17_lista_phase_portraits_20260414/fixed17_h1000_h3000_h5000_lista_phase_portraits_manifest.json).
  It writes `51` per-horizon figures (`17 x 3`) plus per-system selection
  metadata, selecting one best collected LISTA run per system by saved
  `H1000` best-periodic mean and reusing that run's saved `H1000` best-
  periodic mode for all three horizons. Because the packet spans `8` distinct
  LISTA roots rather than only the two promoted finalists, treat it as a
  presentation artifact for handoff and appendix use, not as new causal
  shortlist evidence.
- An April 13 audit confirmed that every paper-critical SLURM packet that
  finished in the last 7 days is now represented in the core docs. The needed
  maintenance in this pass is status cleanup, not new queue discovery: the
  clean tanh / no-shrink control, the `v7` seed-`0` shortlist refinement, and
  the narrow `v6` / `v7` `200k` forecast-floor check are all closed evidence,
  not pending runs.
- On the fixed benchmark systems, evaluate those long-horizon forecasting
  errors as a function of basin depth / separatrix proximity. The expected
  signature is best forecasting deep inside a basin, where the model can
  identify a local linearization cleanly, and worst forecasting near a
  separatrix, where basin identity and the appropriate local chart are most
  ambiguous. This conditioning is benchmark-only evaluation; it should not be
  turned into a training-time assumption.
- Transition-path diagnostics and oracle chart analyses remain supporting
  checks that help explain why a support partition is or is not trustworthy;
  they do not replace the basin-support objective itself.
- Treat the local-linear-law read as basis-aware and symmetry-aware. Two basins
  may share similar eigenvalues while differing mainly in orientation,
  symmetry transform, or encoder chart, so the evaluation target is not
  “different eigenvalues per basin”; it is raw plus aligned operator/Jacobian
  separation together with support-family uniqueness up to alignment.
- The first live LISTA basin-partition sweep on the fixed `17` systems was
  launched at each system's default `dt` under
  [transition_rich_basin_partition_20260407](/network/scratch/l/lia/skae/transition_rich_basin_partition_20260407).
  That already-launched April 7 queue used the older `200000`-step budget and
  should now be treated as legacy long-budget LISTA evidence rather than the
  forward default.
- All new interpretability / metric-diagnosis training on the fixed
  `17`-system shortlist should use `20000` optimization steps. Reserve
  `200000`-step training for the final paper-facing rerun only, after the
  shortlist, metrics, and model recipe are locked.
- All new interpretability / metric-diagnosis training on the fixed
  `17`-system shortlist should use `1` seed by default while screening
  method-side and evaluation-side ideas. Promote a result to `3` seeds only
  after the seed-`0` reduction looks strong enough to be a real paper-facing
  candidate whose robustness needs confirmation.
- All new LISTA comparator sweeps on that fixed shortlist should use the
  default sampling regime so that sampling does not become a confounder in the
  comparison against the MLP controls. Keep hard-init or other altered
  sampling schemes as a separate axis and, once the base recipe is locked,
  report them as separate table entries rather than mixing them into the main
  comparator row.
- The earlier seed-`0`, `20k` shortlist packets remain available on disk under
  [transition_rich_basin_partition_20260409_seed0_smoke_v1](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v1)
  through
  [transition_rich_basin_partition_20260409_seed0_smoke_v6](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v6)
  plus
  [transition_rich_basin_partition_hardinit_seed0_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409),
  so new LISTA iterations can reuse the existing `20k` anchors rather than
  rerunning them.
- One narrow follow-up around the current `v5 p64` shortlist winner is now
  complete under
  [transition_rich_basin_partition_hardinit_seed0_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409).
  This seed-`0`, `68`-task packet was the first forward run of the
  near-separatrix hard-initialization axis on the fixed `17` systems. It keeps
  the same `20k` budget and compares four roots:
  `lista_blockdiag_signsplit_basin_partition`,
  `lista_blockdiag_signsplit_hardinit_basin_partition`,
  `lista_dense_softblock_signsplit_p64_basin_partition`, and
  `lista_dense_softblock_signsplit_p64_hardinit_basin_partition`.
- The hard-init reduction now gives a mixed but useful read. The
  block-diagonal hard-init variant is the clearer interpretability positive:
  at `absolute:0.001` on deep-basin states it improves
  `H(S|B)` (`1.4297 -> 1.3493`), `U_exact` (`0.7181 -> 0.7447`),
  `H(F|B)` (`0.1129 -> 0.1018`), own-basin canonical projection ratio
  (`25.5197 -> 7.7018`), wrong-support ratio (`0.7599 -> 0.3034`), and raw
  operator-family separation (`1.8908 -> 2.4271`) with nearly neutral
  forecasting. The dense `p64` hard-init variant is more mixed on raw support
  compression, but it improves forecasting strongly
  (`H1000` system-median best `0.1358 -> 0.0794`) and also improves the
  deep-basin canonical projection and wrong-support ratios
  (`9.9799 -> 3.0431`, `0.8926 -> 0.6768`).
- That hard-init result should now be treated as a sampling-regime effect first
  and an architecture result second. The paper-facing tables should keep
  standard-sampling and hard-init rows separate, and the fair architecture
  comparison under hard-init needs matched hard-init MLP controls rather than
  only the current standard-sampling controls.
- On the locked fixed-`17`, `200k`, `10`-seed packet, the best forecasting
  root still remains
  `lista_blockdiag_signsplit_hardinit_basin_partition` with system-median
  best-periodic `H1000 = 0.0516`. That is the current best fixed-`17`
  finalized forecasting value from the locked packet, but it should no longer
  be read as evidence that hard-init helps only LISTA because the matched
  hard-init sparse MLP control is now already better on pass `0`.
- The matched hard-init control-only follow-up under
  [results/transition_rich_hardinit_mlp_controls_seed10_20260416](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416)
  is now forecasting-complete through pass `1`. Launcher `9285895`, initial
  array `9285897`, pass-`0` collect / resolve / advance
  `9285898 -> 9285899 -> 9285900`, and rescue pass `1`
  `9291399 -> 9291400 -> 9291401 -> 9291402` all finished. The finalized
  pass-`1` forecasting summary under
  [collect_pass1/forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/collect_pass1/forecasting_summary.md)
  now closes the forecasting-side fairness packet across the three hard-init
  control roots:
  `mlp_sparse_hardinit_basin_partition_control`,
  `mlp_zero_sparse_hardinit_basin_partition_control`, and the newly exposed
  `mlp_sparse_blockdiag_hardinit_basin_partition_control`.
- The finalized pass-`1` forecasting summary sharpens the causal read. The
  sparse hard-init MLP remains the best forecasting control at system-median
  best-periodic `H100/H500/H1000 = 0.0082 / 0.0260 / 0.0273` on `167` rows.
  The structured blockdiag hard-init MLP lands at
  `0.0094 / 0.0359 / 0.0383` on `316` collected rows, so it is competitive and
  still clearly stronger than the zero-sparse tanh / no-shrink control
  (`0.5704 / 2.6733 / 3.8044` on `167` rows), but it does not overtake the
  sparse hard-init MLP forecasting lead. The hard-init forecasting gain
  therefore looks even less architecture-specific than the pass-`0` read
  suggested: induced sparsity still matters sharply, while the strongest
  forecasting win under hard-init is presently a sparse MLP, not LISTA.
- The rescue / `dt` blocker on that packet is now closed. Pass-`1`
  [dt_resolution/pass1/dt_resolution.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/dt_resolution/pass1/dt_resolution.md)
  shows all `51/51` arm-system pairs as `accepted_default`, and
  [advance_pass1.json](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/automation/advance_pass1.json)
  records `request_rows = 0` with no further rescue pass requested.
- The matched-sampling hard-init control packet is now fully reduced. The
  original reducer `9295034` did time out, but the replacement shard jobs
  `9304602-9304604` finished cleanly and the patched merge / summary reruns
  `9304747 -> 9304748` wrote
  [interpretability_final_pass1](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/interpretability_final_pass1)
  with `13,554` rows and `0` failures plus the finalized paper-facing
  comparison in
  [final_comparison_pass1](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1).
  On the selected `absolute:0.001` / `deep` slice, the two sparse hard-init
  MLP controls are almost tied: blockdiag sparse MLP gives
  `0.0082 / 0.0252 / 0.0264` at `H100/H500/H1000`, plain sparse MLP gives
  `0.0082 / 0.0260 / 0.0273`, and both have
  `H(B|S)=0.0000`, `H(S|B)=0.2068`, `U_exact ~= 0.98`, and `H(F|B)=0.0000`.
  The tanh / no-shrink hard-init control remains much worse on forecasting and
  on wrong-support ablation robustness, so induced sparsity still matters more than
  the exact sparse-encoder architecture in this oversampled setting.
- The first paper-facing cross-root comparison pass is now dependency-chained
  behind that live hard-init reducer under
  [transition_rich_post_hardinit_crossroot_eval_20260409](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409).
  Its phase-`1` chain is `9210427` (`collect_tr_crossroot`) ->
  `9210429` (`tr_interp_crossroot`), both submitted with `afterok`
  dependencies so they only run if the live hard-init reducer `9209616`
  completes successfully. The queued bundle is defined explicitly in
  [selected_roots.txt](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409/root_specs/selected_roots.txt)
  and compares the strongest current `v5` roots, the strongest current `v6`
  roots, the matched `v1` sparse MLP control, and the hard-init packet's
  retrained base / variant pairs under one common reduction protocol.
- That originally submitted cross-root reduction remains invalid paper
  evidence: it wrote `0` rows and `17` failures because legacy LISTA
  checkpoints hit the old `encoder.We.*` compatibility path and the launcher
  collapsed `ROOT_LABELS_CSV` to one label. After fixing both issues locally,
  the clean paper-facing rerun under
  [interpretability_rerun_compat_20260409](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409/interpretability_rerun_compat_20260409)
  completed as job `9211252` with `4131` rows and `0` failures. On deep-basin
  states at `absolute:0.001`, `hardinit_packet_blockdiag_hardinit` improves on
  its retrained base in `H(S|B)` (`1.4278 -> 1.3487`), `U_exact`
  (`0.7184 -> 0.7340`), `H(F|B)` (`0.1128 -> 0.1016`), own-basin projection
  ratio (`25.5175 -> 7.7018`), and wrong-support ratio (`0.7589 -> 0.3035`).
  `hardinit_packet_dense_p64_hardinit` remains the stronger forecasting /
  intervention tradeoff with `H(S|B)=0.7952`, `U_exact=0.8161`,
  `H(F|B)=0.0456`, `own/base=3.0430`, and `freeze/base=0.6715`. The matched
  sparse MLP control does not displace either finalist on the branch
  objective (`H(S|B)=1.0922`, `U_exact=0.7757`, `own/base=15.5769`,
  `freeze/base=1.0690`), so the seed-`0` ranking is now locked.
- The branch is now treated as method-locked enough to run only the final
  confirmatory comparison. Based on the now-validated seed-`0` evidence, the
  promoted LISTA roots are
  `lista_blockdiag_signsplit_hardinit_basin_partition` and
  `lista_dense_softblock_signsplit_p64_hardinit_basin_partition`. The
  confirmatory `200k`, `10`-seed, fixed-`17` default-`dt` packet against the
  sparse MLP control plus a zero-`L1` ReLU ablation ran under
  [transition_rich_basin_partition_final_seed10_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409)
  as default array `9211290_[0-679]`, with pass-`0` collect / resolve
  `9211291 -> 9211292`, rescue pass-`1` collect / resolve
  `9214918 -> 9214919`, and final reducer / comparison `9218036 -> 9218037`.
- That final confirmatory packet is now fully finalized through rescue pass
  `1`. The watcher output
  [advance_pass1.json](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/automation/advance_pass1.json)
  reports `finalized: true`, `request_rows: 0`, and `final_pass: 1`, and
  [failures.json](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/interpretability_final_pass1/failures.json)
  is empty.
- The locked multi-seed mixed-regime read now sharpens the branch conclusion
  rather than only confirming forecast viability. On the selected deep-basin
  `absolute:0.001` slice in
  [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md),
  `lista_dense_softblock_signsplit_p64_hardinit_basin_partition` beats the
  standard-sampling sparse MLP control in `H(S|B)` (`0.2449 -> 0.0543`), `U_exact`
  (`0.9772 -> 0.9923`), and `freeze/base@20` (`0.3923 -> 0.1691`), with
  paired wins on `15/17`, `14/17`, and `16/17` systems while staying
  forecast-competitive (`H1000 0.0768` vs `0.0608`). Against the zero-`L1`
  ReLU ablation it also wins `H(S|B)` on `13/17` systems,
  `U_exact` on `12/17`, `freeze/base@20` on `15/17`, and `H1000` on
  `11/17`. On this selected slice `H(F|B)` saturates at `0.0000` for all
  four roots, so the decisive multi-seed discriminators are exact-support
  fragmentation, wrong-support ablation robustness, persistence, and forecasting
  rather than family entropy.
- Because the promoted LISTA roots use hard-init oversampling and the current
  MLP controls do not, that locked packet should be read as architecture plus
  sampling evidence, not as an architecture-isolating causal claim by itself.
- The block-diagonal hard-init root now reads as the forecast-retaining
  companion rather than the main basin-support winner on the locked packet.
  It keeps the best `H1000` (`0.0516`) and better freeze robustness than both
  MLP controls (`freeze/base@20 = 0.2665` vs `0.3923` sparse, `0.7162`
  zero-sparsity), but on the same slice it loses exact-support compression to
  the matched sparse MLP control (`H(S|B)=0.3219` vs `0.2449`,
  `U_exact=0.9646` vs `0.9772`; paired losses `15/17` on both metrics).
- For coauthor-facing writeup, the locked packet should no longer be summarized
  only by a single `H1000` read. The basin-support interpretability branch
  should report long-horizon forecasting jointly at `H100`, `H500`, and
  `H1000`, and the key mechanistic slice should compare deep-basin states
  against near-separatrix states. The branch hypothesis is that sparsity helps
  forecasting when it lets the Koopman state keep track of the active basin;
  if that is true, the gains should be strongest deep in a basin and weakest
  near separatrices.
- A paper-facing paired-comparison summarizer is now ready at
  [summarize_transition_rich_final_comparison.py](/home/mila/l/lia/skae/tools/summarize_transition_rich_final_comparison.py).
  The locked packet has now used it to emit one combined markdown / JSON read
  under
  [final_comparison_pass1](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1),
  covering root-level medians plus per-system paired wins for both LISTA
  finalists against both MLP controls on the selected support slice.
- The new study-plan-aligned state-level reducer is now smoke-validated on the
  historical native-trio LISTA checkpoints under
  [transition_rich_interpretability_smoke_20260409/native_seed0](/home/mila/l/lia/skae/results/transition_rich_interpretability_smoke_20260409/native_seed0),
  and its next diagnostic tier is now also validated under
  [transition_rich_interpretability_smoke_20260409/native_seed0_v2_family_projection](/home/mila/l/lia/skae/results/transition_rich_interpretability_smoke_20260409/native_seed0_v2_family_projection).
  On that legacy `200k`, seed-`0` native subset, deep-basin `H(B|S)` is
  already `0.0000` for both dense and block-diagonal LISTA across several
  support definitions, but `H(S|B)` remains high unless support is forced into
  aggressive top-`k` masks. The new family view sharpens the paper read:
  greedy support-family clustering can collapse that fragmentation strongly
  (`H(F|B) ~= 0.1322` dense, `0.2388` block-diagonal at `absolute:0.001`),
  while the deep-basin canonical exact-support intervention still worsens
  one-step prediction (`own/base > 1`) even though wrong-basin projections are
  catastrophic. The live problem is therefore not basin contamination; it is
  too many exact supports per basin and too little evidence that one canonical
  exact support is itself the operative local chart.
- The earlier forward shortlist packets `v1-v4` are now also complete at their
  submitted scope:
  [transition_rich_basin_partition_20260409_seed0_smoke_v1](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v1),
  [transition_rich_basin_partition_20260409_seed0_smoke_v2](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v2),
  [transition_rich_basin_partition_20260409_seed0_smoke_v3](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v3),
  and
  [transition_rich_basin_partition_20260409_seed0_smoke_v4](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v4).
  `v1` contains the matched sparse MLP control, `v2` covers the first
  HyperLISTA / doubled-block sweep, `v3` covers reset-policy and dynamics-
  aware reencoding, and `v4` covers the first structured / soft-block tier.
- The previously still-missing reset-policy tier is now implemented locally:
  standardized evaluation can warm-start reencoding from the predicted latent,
  run projection-gap or hybrid event-triggered resets with group ambiguity,
  off-block spillover, support-margin fragility, minimum dwell, and maximum
  interval controls, save rollout reset diagnostics, and expose a
  `best_reset` summary alongside the older `best_periodic` read. The richer
  trigger ablations are now runnable through the transition-rich task
  manifest, but they remain scientifically open because they have not yet
  been rerun on the fixed shortlist. Focused validation is complete: `75`
  targeted tests passed on a compute node, and a CLI smoke run with the new
  evaluation flags successfully emitted `event_proj_0p05` artifacts under
  [/tmp/skae_eval_smoke/20260409-011940](/tmp/skae_eval_smoke/20260409-011940).
- The next encoder-side interpretability tier is now also implemented locally:
  LISTA and HyperLISTA can infer latent groups from block-diagonal, soft-
  block, or structured layouts, apply sparse-group shrinkage across those
  groups, and optionally keep only the top-`k` groups before within-group
  thresholding. The transition-rich manifest now includes shortlist variants
  for block-diagonal sparse-group LISTA, soft-block sparse-group LISTA, and a
  block-diagonal HyperLISTA top-`2` group-selection ablation. The
  block-diagonal sparse-group arm has now been screened in `v7`; the
  soft-block sparse-group and HyperLISTA top-`2` arms remain the unrun pieces
  of this tier on the fixed shortlist.
- The next LISTA method-side tier after that is now also implemented locally:
  standard LISTA can use sample-dependent thresholds driven by reconstruction
  residual and latent-prior mismatch, learn separate base thresholds per
  inferred latent group, and swap its pre-code between free-MLP, linear,
  dictionary-tied, and hybrid tied-plus-residual modes. The transition-rich
  manifest now includes runnable shortlist variants for a block-diagonal
  adaptive/groupwise-threshold LISTA arm and dense soft-block
  dictionary-tied / hybrid-precode arms. Those variants have now all been
  screened in `v7`, so this tier is no longer a plumbing or launch gap; it is
  negative / mixed shortlist provenance.
- The next still-open design-note axis after that is also implemented locally:
  standard LISTA now supports fixed-beta momentum refinement, and the
  transition-rich manifest includes runnable sign-split momentum variants on
  the current block-diagonal forecast-retaining root and the dense soft-block
  `p64` root. Those variants have now been screened in `v7` and did not
  displace the locked finalists, so momentum is no longer an unrun shortlist
  gap.
- The stronger soft-block sweep that the design note still called out is now
  also exposed locally on the paper-facing sign-split `p64` root: the
  transition-rich manifest now includes higher-weight dense soft-block
  variants at `5e-4` and `1e-3` in addition to the earlier `1e-4` default.
  Those variants have now also been screened in `v7`. The `5e-4` arm gives
  the best new deep-basin support-compression read in that packet, but it
  misses `1/17` systems and does not dominate forecasting, so the soft-block
  question is now evidence rather than manifest coverage.
- The next design-note tier after `v3` is now implemented and smoke-validated
  locally: the repo supports dense soft block-sparse `K` penalties, structured
  hard-block LISTA packets, and reducer-side dominant-group metrics that
  respect both structured global-plus-basin layouts and soft-block partitions.
  Validation completed with `98` targeted tests passing on a compute node,
  plus end-to-end CLI smokes under
  [/tmp/skae_softblock_smoke/20260409-014101](/tmp/skae_softblock_smoke/20260409-014101)
  and
  [/tmp/skae_structured_smoke/20260409-014101](/tmp/skae_structured_smoke/20260409-014101).
- `v4` still carries a launcher-level failure record because `9202903` hit
  `AssocMaxSubmitJobLimit`, but every emitted default / rescue / reducer job
  completed successfully, so the submitted `v4` scope is scientifically
  closed even though the launcher itself is not clean.
- The six forward fixed-`17` interpretability packets `v1-v6` are now closed
  through every submitted collect / resolve / interpretability-reduce stage.
- The completed `v5` reduction is the strongest new exact-support positive so
  far. On deep-basin states at `absolute:0.001`,
  `lista_dense_softblock_signsplit_p64_basin_partition` reaches
  `mean H(S|B)=0.7719`, `mean U_exact=0.8064`, and `mean H(F|B)=0.0521` while
  still keeping `16/17` systems under the `H1000` good-forecast gate
  (`system-median best-periodic = 0.1819`). The best forecast-retention root
  in the same packet is `lista_blockdiag_signsplit_basin_partition` at
  `H1000 system-median = 0.0119` with `17/17` good systems, but its exact-
  support compression is weaker than the `p64` dense root.
- The completed `v6` reduction clarifies the next tradeoff rather than
  delivering a new overall winner. Coherence-only roots preserve or improve
  forecasting most strongly
  (`lista_dense_softblock_signsplit_coherence_basin_partition` reaches
  `H1000 system-median = 0.0585` with `17/17` good systems), while
  linear-encoder roots reduce within-basin support fragmentation
  (`lista_dense_softblock_signsplit_linear_encoder_coherence_basin_partition`
  reaches `mean H(S|B)=1.0575`, `mean U_exact=0.7837`, `mean H(F|B)=0.0841`)
  but at a large forecasting cost (`H1000 system-median = 0.9118`,
  `15/17` good systems). `v6` therefore does not displace `v5 p64` on the
  current forecast-versus-interpretability frontier.
- `v5` should now be read as the sign-split shortlist tier:
  hard block-diagonal and dense soft-block LISTA families with sign-split
  codes, `2` versus `4` LISTA refinement loops, doubled block-count variants,
  and latent-size sweeps including `p=64` and `p=128`.
- `v6` should now be read as the identifiability follow-up to `v5`:
  the same sign-split shortlist families with either a restrained linear
  pre-code, a decoder-coherence penalty, or both. In this branch, coherence
  means penalizing off-diagonal similarity among normalized decoder atoms so
  the decoder dictionary has fewer redundant atoms and therefore fewer
  interchangeable exact supports.
- Use the standard MLP encoder control as the main causal comparator for this
  branch. The locked `200k`, `10`-seed packet already answers the LISTA-vs-
  sparse-MLP comparison on the same basin-separation terms. Its additional
  zero-`L1` arm remains scientifically useful supporting context because that
  task table used `config_name=generic_sparse` with `sparsity_coeff=0.0`. It
  therefore isolates explicit-penalty removal while retaining ReLU-induced
  shrinkage. That makes it a ReLU-only sparsity ablation, not the clean
  no-shrink anti-sparsity control.
- A new working-budget zero-sparsity no-shrink control screen is also now
  complete under
  [transition_rich_zero_sparse_control_noshrink_20k_seed3_20260410](/home/mila/l/lia/skae/results/transition_rich_zero_sparse_control_noshrink_20k_seed3_20260410).
  On the fixed `17` systems at default `dt`, it keeps `17/17` systems inside
  the `H1000 < 50` gate, but its system-median best-periodic forecasting is
  much weaker than the locked finalists (`H100/H500/H1000 = 1.8317 / 3.5797 /
  4.1857`). Treat it as supporting zero-sparsity context rather than as a new
  paper-facing contender.
- The older locked-budget zero-sparsity expansion under
  [transition_rich_zero_sparse_control_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_zero_sparse_control_seed10_20260410)
  as array `9221521_*` should now be read carefully rather than discarded. Its
  task table uses `config_name=generic_sparse`, so it is not the requested
  tanh / no-shrink control. But it is still scientifically meaningful as a
  locked-budget ReLU-only ablation that removes the explicit `L1` penalty
  while keeping architectural shrinkage in the encoder. That packet now runs
  through collect / resolve pass `4`, still accepts default `dt` on all
  `17/17` systems, and remains at system-median best-periodic
  `H100/H500/H1000 = 0.5764 / 2.0556 / 2.6532`.
- The corrected locked-budget tanh / no-shrink control is now complete under
  [transition_rich_zero_sparse_tanh_control_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_zero_sparse_tanh_control_seed10_20260410)
  via wrapper `9224111`. It uses a fresh results tag so the generated task
  table will be rebuilt from the corrected manifest mapping
  (`mlp_zero_sparse_basin_partition_control -> generic_no_shrink`) instead of
  inheriting the stale `generic_sparse` packet. The default pass completed as
  `9224263_* -> 9224264 -> 9224265`; despite one fast array failure, the
  collector wrote `169` rows, accepted default `dt` on all `17/17` systems,
  and reports system-median best-periodic
  `H100/H500/H1000 = 0.5763 / 1.7924 / 2.4279`. This is now the clean
  locked-budget no-shrink control and it remains much weaker than the
  promoted sparse LISTA roots.
- The narrow LISTA-only seed-`0`, `20k`, default-sampling refinement under
  [transition_rich_basin_partition_20260410_seed0_smoke_v7](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260410_seed0_smoke_v7)
  is now complete as `9226564_[0-135] -> 9226565 -> 9226566 -> 9226567` with
  `0` reducer failures. The best new forecasting root is
  `lista_blockdiag_sparsegroup_basin_partition`
  (`H1000 system-median best = 0.0846`), while the strongest new deep-basin
  support-compression read is
  `lista_dense_softblock_signsplit_p64_softblock5em4_basin_partition`
  (`H(S|B)=0.6795`, `U_exact=0.8453`, `H(F|B)=0.0634` at
  `absolute:0.001` / `deep`), but that root is missing `1/17` systems after a
  fast training failure. No `v7` root cleanly dominates both forecasting and
  basin-support alignment, so this packet does not reopen the locked
  shortlist.
- One narrow default-sampling long-budget forecasting follow-up is now
  complete under
  [transition_rich_basin_partition_v6_v7_200k_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_basin_partition_v6_v7_200k_seed10_20260410).
  Its fair default-`dt` comparison completed as array / collector
  `9228394_[0-339] -> 9228395`. The wrapper `9228393` shows `FAILED` only
  because its attempted model-wise dt-rescue continuation both violated the
  matched-`dt` comparison rule and hit `AssocMaxSubmitJobLimit` after the
  default task table had already been written
  ([queue-transition-rich-dt-9228393.err](/network/scratch/l/lia/skae/queue-transition-rich-dt-9228393.err)).
  The actual fair run is the completed default array plus collector, which
  wrote `329/340` rows across both roots and all `17` systems
  ([collect-transition-rich-9228395.out](/network/scratch/l/lia/skae/collect-transition-rich-9228395.out)).
- That long-budget default-sampling check is negative for shortlist reopening.
  The better of the two roots,
  `lista_dense_softblock_signsplit_coherence_basin_partition`, reaches
  system-median best-periodic `H100/H500/H1000 = 0.0416 / 0.0761 / 0.0796`,
  while `lista_blockdiag_sparsegroup_basin_partition` lands at
  `0.0437 / 0.1142 / 0.1193`. The coherence root is still slightly worse than
  the promoted dense hard-init finalist at all three horizons
  (`0.0196 / 0.0733 / 0.0775`) and also worse than the matched sparse MLP
  control on the long horizons that matter most (`0.0614 / 0.0608` at
  `H500/H1000`), so this forecasting-only packet does not displace the locked
  basin-support finalists or add a new interpretability reduction.
- Outstanding paper-critical problem:
  the locked multi-seed comparison already resolves the LISTA-vs-sparse-MLP
  part of the question on the selected deep `absolute:0.001` slice: the dense
  `p64` hard-init LISTA root beats the matched sparse MLP control on
  exact-support fragmentation and wrong-support ablation robustness while remaining
  forecast-competitive, whereas the block-diagonal hard-init root survives as
  the forecast-retaining companion rather than the lead exact-support result.
  The clean tanh / no-shrink `200k`, `10`-seed control now lands the broader
  induced-sparsity contrast as well: removing induced sparsity leaves
  forecasting much weaker than the promoted sparse LISTA roots even when
  `17/17` systems pass the default-`dt` gate. The narrow default-sampling
  `200k` forecast-floor check on the best `v6` / `v7` roots is now also
  complete and negative, so the shortlist should not be reopened. The
  remaining paper-critical risk is now the state-conditioned
  `H100/H500/H1000` read showing whether forecasting is strongest deep in
  basin and weakest near separatrices, plus the final calibration question of
  whether the main text should make a selected-slice exact-support-reuse claim
  centered on the dense `p64` root or a more conservative family/group or
  symmetry-aware alignment claim because `H(F|B)` saturates on that slice.
- Do not treat stronger chart-switch-localization or sparse-only mechanism
  claims as the acceptance criterion for this branch. Those questions remain
  useful supporting context, but the live branch succeeds or fails on whether
  LISTA yields cleaner basin-separated support structure than the matched
  standard MLP encoder on the fixed shortlist.
- The planning-doc gap is now narrower and explicit. The main unresolved
  method-side fairness question is matched-sampling hard-init MLP controls
  plus still-unrun symmetry-aware tying on symmetric toys. The previously
  unrun LISTA shortlist axes from the design note are no longer configuration
  gaps: stronger soft-block penalty sweeps, momentum LISTA, adaptive or
  blockwise LISTA thresholds, dictionary-tied / hybrid pre-codes, group-aware
  sparse-group shrinkage / group-first support selection, and richer reset
  triggers beyond projection-gap have now all been at least screened on the
  fixed shortlist, and the long-budget default-sampling `v6` / `v7` follow-up
  did not promote any of them over the locked finalists. The main unrun
  evaluation-side
  items are wrong-support interventions, controlled-transfer switch-timing
  metrics, state-conditioned long-horizon forecasting at `H100/H500/H1000`
  split by basin depth / separatrix proximity, basis-aware support-conditioned
  Jacobian or operator-family analyses, and the paper visual-diagnostic suite.
- The state-level interpretability reducer now also has local code support for
  those evaluation-side study items: canonical wrong-support rollout metrics,
  first-switch timing summaries, sampled effective-Jacobian family summaries,
  and optional support-family visual artifacts (phase portraits, entropy maps,
  switch rasters, basin/support confusion, operator-distance heatmaps). Those
  diagnostics are implemented but not yet rerun on the fixed `17` shortlist,
  so they remain scientifically open as evidence rather than as missing
  tooling. One important caveat after the April 9 study-definition update:
  the reducer already supports the **raw** operator-family and Jacobian-family
  summaries, but it does **not** yet implement the new similarity-aligned
  operator distances, eigendirection comparisons, or invariant-subspace-angle
  metrics. Those basis-aware diagnostics remain a tooling gap rather than an
  execution gap.
- The overnight April 8 continuation closes the practical step-size question
  more broadly than the earlier native-trio manual audit:
  [collect_pass0](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/collect_pass0/forecasting_summary.md)
  already reports `17/17` good systems at `H1000` for both dense and
  block-diagonal LISTA on the fixed shortlist, and the six completed overnight
  `20k` rerun waves keep all `16/16` rerun systems below the same gate at the
  same default per-system `dt`.
- Those overnight jobs should not be interpreted as evidence that smaller `dt`
  rescues were needed or helpful. The collector had been dropping `env_dt` for
  `gated_*` and `claude:*` arms, so the resolver kept reissuing the same
  default-`dt` tasks. The collector is now fixed locally; the scientific
  takeaway from the overnight work is stronger default-`dt` adequacy, not a
  smaller-`dt` effect.
- The first full fixed-`17` LISTA basin-support reduction is now complete
  under
  [basin_support_metrics_20260408_v3](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/basin_support_metrics_20260408_v3):
  - mean support-group purity is `0.9883` for block-diagonal LISTA and
    `0.9961` for dense LISTA
  - mean retained-trajectory coverage is `0.8729` for block-diagonal LISTA and
    `0.8787` for dense LISTA, with `15/17` systems above the `0.60` coverage
    gate for both roots
  - the local support-conditioned `H=20` fit does **not** beat the matched
    global fit on any of the `34` LISTA runs, and it beats the shuffled
    baseline only once (`dense x duffing_triple_well`) where coverage is only
    `0.1172`
- Exact metric definitions and caveats for that reduction now live in
  [docs/transition_rich_basin_support_metric_definitions.md](/home/mila/l/lia/skae/docs/transition_rich_basin_support_metric_definitions.md).
- The current LISTA packet therefore supports a basin-pure recurring-support
  claim much more strongly than a local-linear-mechanism claim. Even the best
  native positives now read as purity / coverage positives rather than as
  `H=20` local-linearity wins.
- The deterministic `2D` toy suite is locally implemented and calibrated:
  `multiwell_strong_transition`, `gated_local_linear`,
  `gated_transfer_linear`.
- Keep the fair `200k` benchmark packet, the hard-system packet, and the older
  mechanism packet only as supporting evidence rather than as the live branch
  definition.
- Historical design inventories, broader Claude-catalog screens, and older
  non-shortlist mechanism negatives are supporting provenance only. They are no
  longer the live branch definition.
- For design choices inside the next interpretability-ablation loop over plan
  items `3` and `4`, treat
  [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md)
  as the current ground-truth planning note, but do not treat those design
  choices as settled evidence until the corresponding experiments are run and
  written back into the live docs.

What stays live here:
- Fixed-`17` basin-separation evidence for dense-LISTA and block-diagonal
  LISTA.
- The matched standard-MLP control needed to test whether LISTA gives stronger
  basin separation than a non-LISTA encoder on the same systems.
- Native-trio and Claude-subset diagnostics that measure basin purity, support
  reuse, support-view clustering, and local predictive structure.
- Benchmark-only long-horizon forecasting diagnostics at `H100`, `H500`, and
  `H1000` that test whether forecasting is strongest deep inside a basin and
  weakest near a separatrix.
- Only the benchmark, hard-system, and older mechanism packets needed to
  position the branch in the broader paper.

Outstanding problem:
- No toy-system design blocker remains, and system selection is no longer
  open. The active interpretability branch is frozen to the `17` systems listed
  above. The full LISTA reduction now exists and says recurring supports are
  usually basin-pure and broad-coverage, but they do not yield a stronger
  support-conditioned `H=20` local-linear fit than one global map on this
  packet. The new state-level native-trio smoke reduction sharpens that same
  read: supports can already be basin-pure deep inside a basin without giving
  one exact support per basin. The lead blocker is therefore no longer the
  LISTA reduction itself, but the fair architecture comparison is still not
  closed: the locked packet gives a useful mixed-regime LISTA-versus-MLP read,
  yet the promoted LISTA roots use hard-init oversampling while the completed
  MLP controls in that packet are standard-sampling. The matched hard-init
  control packet under
  [results/transition_rich_hardinit_mlp_controls_seed10_20260416](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416)
  is now forecasting-finalized and already says the hard-init forecasting gain
  is not LISTA-exclusive: the sparse hard-init MLP control reaches
  `H100/H500/H1000 = 0.0082 / 0.0260 / 0.0273`, the structured blockdiag
  hard-init MLP reaches `0.0094 / 0.0359 / 0.0383`, and both beat the locked
  hard-init LISTA forecasting winner at `H1000` (`0.0516`), while the clean
  zero-sparse hard-init control is much worse. The architecture-isolating
  hard-init read is now closed at the artifact level: the matched hard-init
  interpretability packet and paired final comparison both exist, and they say
  the two sparse hard-init MLP controls are nearly tied on the selected deep
  slice while the zero-sparsity control is much weaker functionally. The live
  blocker is therefore no longer queue completion or missing hard-init
  evidence; it is claim calibration. Keep standard-sampling and hard-init rows
  separate in the paper tables, decide how prominently to foreground the
  blockdiag hard-init MLP as a forecast-retaining companion result, and make
  explicit that `H(S|B)` alone is not the whole causal read because the
  zero-sparse control is not uniformly worse on that entropy metric even while
  it degrades forecasting and wrong-support ablation robustness. On the supporting Dysts
  benchmark, the live execution blocker is closed: the blockdiag-MLP
  long-horizon extension under
  [results/dysts_long_horizon_eval_mlp_blockdiag_20260415](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_mlp_blockdiag_20260415)
  is now complete, so the remaining Dysts work is paper positioning and clear
  presentation of the seven-root comparison rather than more queue recovery.

Assumption split:
- Training/deployment target: basin count and basin labels are unknown.
- Benchmark evaluation: known endpoint-basin counts and labels are allowed for diagnostics.

## Paper-Facing Experiment Protocol

1. Decide whether a result belongs in this live file before running or writing it down.
   - Keep it live only if it directly supports one of the four evidence
     buckets in
     [PAPER_EXPERIMENT_EVIDENCE_MAP.md](/home/mila/l/lia/skae/docs/PAPER_EXPERIMENT_EVIDENCE_MAP.md),
     or if it is the newest paper-critical execution update.
   - Move appendix-only tuning, queue chronology, and superseded subthreads to `docs/EXPERIMENTS_ARCHIVE.md`.
2. Define the causal test before queueing.
   - Write the objective or claim, baselines and fairness controls, exact systems, seeds, horizons, metrics, acceptance criteria, failure criteria, and output roots.
   - For new toy systems, define `endpoint basin`, `transition`, basin-count target, crossing-fraction gate, deterministic mechanics, and calibration outputs before implementation.
3. Keep the paper fairness rules fixed.
   - Keep the frozen benchmark and hard-system supporting packets on the
     paper `200k` budget.
   - For the live interpretability / metric-design loop on the fixed
     `17`-system shortlist, use `20000` training steps by default and reserve
     `200000` only for the final confirmatory paper rerun after the shortlist,
     metrics, and model recipe are locked.
   - For that same live interpretability loop, use `1` seed by default during
     diagnostic screening. Expand to `3` seeds only after a shortlisted effect
     looks strong enough to justify a paper-facing robustness check.
   - When a run has multiple seeds, report medians across seeds, not
     best-seed results.
   - Use the official checkpoint rule from `evaluation_results_best.json`.
   - Do not rely on basin labels or known basin counts when proposing training-time methods.
   - For paper-facing comparisons that claim an architecture effect, keep the
     training sampling regime identical across compared roots. If hard-init
     oversampling is used, compare against controls trained with the same
     sampling regime and report standard versus hard-init as separate table
     entries.
   - For basin-support interpretability packets, pre-register the long-horizon
     forecasting read at `H100`, `H500`, and `H1000`; do not let a lone
     `H1000` summary stand in for the full mechanism test.
   - On benchmark systems, report those long-horizon forecasting metrics both
     overall and stratified by basin depth / separatrix proximity. The target
     signature is best forecasting deep in a basin and worst forecasting near
     a separatrix. Treat that stratification as evaluation-only.
   - New deterministic toy systems must target `3-10` endpoint basins.
   - For the frozen first-pass pair, use the endpoint-conditioned crossing gate in the acceptable `0.30-0.70` range.
   - For the explicit chart-switching transfer family, use source-neighborhood transfer fractions together with inner-core retention rather than the old endpoint-conditioned crossing gate.
4. QA before queueing.
   - Run Python entry points with `uv run`.
   - Write and run the tests first before adding any system-specific environment code or metric code.
   - Smoke-test task builders and collectors locally.
   - Check task counts, seed coverage, root labels, line endings, and `sbatch --export` arguments before launch.
   - Submit SLURM jobs with `sbatch` on the `long` partition unless a different partition is explicitly justified.
5. Report results in this order.
   - Concrete result(s).
   - Result in experimental context.
   - Interpretation.
   - Project implications.
   - Next steps.
6. Update the docs in the same pass.
   - Refresh `Current Status Summary`, `Outstanding problems`, `Queue Status`, and the relevant live section in this file.
   - Update `docs/PAPER_TRACK_STATUS.md` whenever the result changes paper positioning, claims, or wrap-up priorities.
   - Archive detailed queue-era or lower-priority material in `docs/EXPERIMENTS_ARCHIVE.md` instead of letting this file grow again.

### 2026-04-14: Dysts H5000 phase-portrait packet from strongest H3000-ranked LISTA checkpoints

1. Concrete results:
   job `9269340` completed on `main-cpu` and ran
   [tools/generate_dysts_h5000_phase_portraits.py](/home/mila/l/lia/skae/tools/generate_dysts_h5000_phase_portraits.py),
   writing
   [docs/figures/dysts_phase_portraits/dysts_h5000_lista_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h5000_lista_phase_portraits_manifest.json)
   plus `15` per-system PNG/PDF/JSON packets. Under the script's
   H3000-shortlist then H5000-rescore protocol, all `15/15` final selections
   came from dense LISTA. Shared-batch H5000 best-periodic means range from
   `0.0071` on `dysts:SprottTorus` to `4.3022` on `dysts:Duffing`, with
   median `0.6751`; selected periodic modes are `{20:1, 40:2, 80:4, 200:7,
   300:1}`.
2. Result in experimental context:
   this was a figure-generation pass, not a new training sweep. It reused the
   current matched paper-facing Dysts LISTA roots, ranked checkpoints per
   system by saved `H3000` best-periodic mean from
   `evaluation_results_best.json`, rescored the top shortlist at `H5000`
   using each run's saved `H3000` periodic mode, and plotted the best H5000
   rollout against ground truth.
3. Interpretation:
   the strongest current long-horizon Dysts visual packet is fully dominated
   by the dense LISTA family under this selection rule, so the appendix /
   handoff visuals now align with the already-promoted dense LISTA supporting
   benchmark lineage rather than splitting across dense and block-diagonal
   LISTA roots. This is a visualization outcome, not a new causal
   branch-level result.
4. Project implications:
   the senior-coauthor handoff no longer needs a follow-up queue just to fill
   the Dysts long-horizon visual appendix. The remaining work is curation:
   decide which subset of these `15` figures belongs in the handoff packet,
   appendix, or slides, and keep them explicitly secondary to the fixed-`17`
   basin-support branch.
5. Next steps:
   add a compact selector note for the representative Dysts figures to
   include in the handoff materials, then cite the manifest as the source of
   truth for the full visual appendix and do not reopen the checkpoint
   selection unless a benchmark family changes.

### 2026-04-14: Dysts H20000 phase-portrait packet from strongest H3000-ranked LISTA checkpoints

1. Concrete results:
   job `9269661` completed on `main-cpu` and ran
   [tools/generate_dysts_h5000_phase_portraits.py](/home/mila/l/lia/skae/tools/generate_dysts_h5000_phase_portraits.py)
   with `--horizon 20000`, writing
   [docs/figures/dysts_phase_portraits/dysts_h20000_lista_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h20000_lista_phase_portraits_manifest.json)
   plus `15` per-system PNG/PDF/JSON packets. The H20000 rescore selected
   dense LISTA on `13/15` systems and block-diagonal LISTA (`sc=6e-3`) on
   `2/15` systems (`dysts:Duffing`, `dysts:WangSun`). Shared-batch H20000
   best-periodic means range from `0.5229` on `dysts:RikitakeDynamo` to
   `68.2272` on `dysts:Duffing`, with median `1.4783`; selected periodic
   modes are `{1:1, 20:1, 40:2, 80:1, 100:2, 200:5, 300:3}`.
2. Result in experimental context:
   this reused the same H3000-shortlist protocol as the H5000 packet, but
   pushed the shared-batch rescore and phase portraits out to `H20000`. The
   script was generalized in the same pass so the target horizon now controls
   filenames, metadata, plot titles, and manifest naming instead of hardcoding
   `h5000`.
3. Interpretation:
   the longer horizon changes the selected family mix. Dense LISTA still
   dominates overall, but the best H20000 visual packet is no longer
   all-dense: `Duffing` and `WangSun` now favor the stronger block-diagonal
   LISTA root. That makes the ultra-long-horizon appendix visually more mixed
   than the H5000 packet and is the right read to preserve for handoff.
4. Project implications:
   the senior-coauthor packet can now show both a medium-long horizon
   (`H5000`) and an ultra-long horizon (`H20000`) without another queue
   pass. It also gives a concrete example where the visually strongest very
   long-horizon root is not always the same family that wins at the shorter
   appendix horizon.
5. Next steps:
   decide whether the handoff should present paired `H5000/H20000` portraits
   for a small representative subset, especially `Duffing` and `WangSun`,
   and keep the manifest pair as the source of truth for any appendix or
   slide curation.

### 2026-04-14: Queued seed-10 Dysts long-horizon reevaluation at H5000/H10000/H20000/H30000

1. Concrete results:
   launcher `9273653` completed on `main-cpu` and wrote the task table
   [dysts_long_horizon_tasks.tsv](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/task_tables/dysts_long_horizon_tasks.tsv),
   task summary
   [dysts_long_horizon_tasks_summary.json](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/task_tables/dysts_long_horizon_tasks_summary.json),
   and queue record
   [queue_record.json](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/queue/queue_record.json).
   The manifest covers `750` tasks = `5` verified benchmark roots x `15`
   Dysts systems x seeds `0-9`, with `0` missing runs. The queued chain is
   cache prebuild `9273655`, dependency-gated validation `9273656`,
   full reevaluation array `9273657`, and collector `9273658`.
   A standalone validation run `9273670` failed immediately and usefully:
   cached trajectories were one step too short for `H30000` sequence windows.
2. Result in experimental context:
   unlike the earlier H5000/H20000 visual packets, this is the first
   all-seeds, all-five-roots Dysts long-horizon benchmark pass. It does not
   shortlist checkpoints by saved `H3000`; it reevaluates every seed/run on
   its native Dysts system once out to `H30000` and reuses that same rollout
   for all four reported horizons.
3. Interpretation:
   there are no new model-comparison MSE results yet because the collector has
   not run, but the benchmark coverage question is now resolved: every
   intended run exists and the reevaluation packet is auditable. The only
   execution bug found so far was infrastructure-side cache length, not
   model-side instability, and it is now fixed in
   [skae/data.py](/home/mila/l/lia/skae/skae/data.py).
4. Project implications:
   once this queue clears, the paper will have a single seed-`10` source of
   truth for Dysts MSE at `H5000/H10000/H20000/H30000`, plus compact
   selected-mode rollout caches under each run directory for later
   phase-portrait and appendix figure generation without repeat inference.
5. Next steps:
   let cache prebuild `9273655` clear, confirm validation `9273656` on the
   patched cache path, and then read the collector outputs under
   [results/dysts_long_horizon_eval_20260414/collect](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/collect)
   before promoting any new Dysts long-horizon table or headline claim.

### 2026-04-15: Collected partial seed-10 Dysts long-horizon benchmark results

1. Concrete results:
   the queue chain finished through collection. Cache prebuild `9273655`,
   validation `9273656`, collector `9273658`, and replacement validation
   probe `9273675` all completed; the main reevaluation array `9273657`
   finished mixed with many `FAILED` tasks. The collector outputs are
   [forecasting_rows.csv](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/collect/forecasting_rows.csv),
   [pending_rows.csv](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/collect/pending_rows.csv),
   [summary.json](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/collect/summary.json),
   and
   [summary.md](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/collect/summary.md).
   Coverage is only `236/750` complete tasks. Root completion is sparse MLP
   `73/150`, zero-sparse MLP `0/150`, dense LISTA `73/150`, block-diagonal
   LISTA `sc=3e-3` `45/150`, and block-diagonal LISTA `sc=6e-3` `45/150`.
   Aggregate medians of per-system medians across completed rows are:
   sparse MLP `0.2595/1.4572/3.4287/3.7608`,
   dense LISTA `0.0568/1.0454/2.6552/3.2430`,
   block-diagonal LISTA `sc=3e-3` `0.6049/1.9807/2.6270/2.8747`, and
   block-diagonal LISTA `sc=6e-3` `0.6573/1.5841/1.9259/2.3002` at
   `H5000/H10000/H20000/H30000`. Horizon-wise system wins are
   `H5000: dense LISTA 10, sparse MLP 3, blkdiag 3e-3 1, blkdiag 6e-3 1`;
   `H10000: dense LISTA 12, sparse MLP 1, blkdiag 3e-3 1, blkdiag 6e-3 1`;
   `H20000/H30000: blkdiag 6e-3 7, blkdiag 3e-3 5, dense LISTA 3`.
2. Result in experimental context:
   this is the first all-roots Dysts benchmark pass that reevaluates each run
   once to `H30000` and reuses that rollout for all four horizons. The packet
   therefore answers the pipeline question, but not yet the final benchmark
   question, because the completed rows are only a partial subset of the
   planned `10` seeds per root.
3. Interpretation:
   among the completed rows, dense LISTA is strongest at the shorter two long
   horizons, while block-diagonal LISTA `sc=6e-3` becomes strongest at the
   ultra-long horizons. The system-level read is mixed rather than uniform:
   `Duffing` is won by block-diagonal `sc=3e-3` at all four horizons,
   `LuChenCheng` by block-diagonal `sc=6e-3` at all four horizons, and
   `QiChen`, `RikitakeDynamo`, and `SprottTorus` by dense LISTA at all four
   horizons. However, zero-sparse MLP has no usable rows at all, so the
   architecture comparison is still incomplete.
4. Project implications:
   the benchmark now has a reproducible collector packet and reusable rollout
   caches for every completed row, so later Dysts phase portraits do not need
   repeat inference for those runs. But the seed-`10` benchmark is not yet
   strong enough to headline the paper or even a benchmark table because the
   missing-task pattern is highly structured rather than random.
5. Next steps:
   inspect and rescue the failed `9273657` tasks, prioritizing the entirely
   missing zero-sparse MLP root and the missing seed tails in sparse MLP and
   dense LISTA, rerun the collector, and only then decide whether the Dysts
   long-horizon benchmark is strong enough for a main-text or handoff-facing
   claim.

### 2026-04-15: Fixed checkpoint-config compatibility and launched Dysts rescue pass 1

1. Concrete results:
   the main failure mode in `9273657` was not scheduler noise or cache length;
   it was backward-incompatible checkpoint config loading. Failed logs under
   `/network/scratch/l/lia/skae/dysts-long-eval-9273657_*.err` consistently
   raised `CompetitiveLVConfig.__init__() got an unexpected keyword argument
   'SYSTEM_SEED'` while deserializing legacy checkpoint configs. The fix is in
   [skae/config.py](/home/mila/l/lia/skae/skae/config.py): `Config.from_dict()`
   now filters unknown dataclass keys before constructing nested config
   objects. A previously failing zero-sparse MLP reevaluation
   (`dysts:Chua`, seed `0`) now completes cleanly and writes
   [evaluation_results_checkpoint.json](/network/scratch/l/lia/skae/paper_zero_sparse_benchmark_200k_20260321/paper_zero_sparse_benchmark/generic_sparse_sc0_ns200k_best/dysts_Chua/dt_0p0002847474579095888/seed_0/20260321-152929/reeval_dysts_long_horizon_h5000_h10000_h20000_h30000/evaluation_results_checkpoint.json).
   Rescue array `9278881` is now queued on `long-cpu` for the stale
   `514`-task pending set captured in
   [pending_rows.csv](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/collect/pending_rows.csv),
   with recollect `9278882` dependency-chained after it. Queue metadata is in
   [rescue_pass1_queue_record.json](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/queue/rescue_pass1_queue_record.json).
2. Result in experimental context:
   this turns the Dysts packet from a passive partial read into an active
   rescue campaign. The cache-length fix in [skae/data.py](/home/mila/l/lia/skae/skae/data.py)
   was necessary for `H30000`, but it was not sufficient: many older paper
   checkpoints still needed schema-tolerant config loading before reevaluation
   could even start.
3. Interpretation:
   the structured missingness pattern in the first collector now has a concrete
   explanation. The sparse MLP seed tails, the entire zero-sparse MLP root,
   and many later LISTA seeds were not measuring worse models; they were never
   entering rollout evaluation because config reconstruction failed up front.
   The successful local reeval of a previously failing zero-sparse task is the
   decisive check that the compatibility patch addresses the real blocker.
4. Project implications:
   if `9278881` clears as expected, the next collector should materially change
   the benchmark coverage and may reopen the zero-sparse comparator row that is
   currently absent. Until that recollect lands, the partial `236/750` packet
   remains informative but not decision-grade.
5. Next steps:
   let `9278881` run, inspect early task logs for any new post-compatibility
   failure modes, and read the refreshed collector from `9278882` before
   updating any benchmark table or paper-facing claim.

### 2026-04-15: Dysts rescue pass 1 completed and refreshed collector to 750/750

1. Concrete results:
   rescue array `9278881` and recollect `9278882` both completed cleanly.
   The refreshed collector summary in
   [summary.md](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/collect/summary.md)
   now reports `750/750` complete tasks and `0` pending or invalid tasks.
   Root-level median best-periodic MSE at `H5000/H10000/H20000/H30000` is:
   sparse MLP `0.1953/1.2373/3.2524/3.6981`,
   zero-sparse MLP `0.2474/1.4564/3.2354/3.7893`,
   dense LISTA `0.1285/0.9778/2.7403/3.2662`,
   block-diagonal LISTA `sc=3e-3` `0.6575/2.0255/2.8856/3.2833`,
   and block-diagonal LISTA `sc=6e-3` `0.6473/1.4534/1.9150/2.2720`.
2. Result in experimental context:
   this closes the benchmark packet that was meant to provide one uniform
   seed-`10` source of truth for Dysts long-horizon forecasting out to
   `H30000`. The rescue pass converted the earlier structured missingness into
   a complete collector without changing the evaluation protocol itself.
3. Interpretation:
   the full benchmark keeps the same qualitative pattern suggested by the
   partial packet but with complete coverage: dense LISTA is strongest at
   `H5000/H10000`, while block-diagonal LISTA `sc=6e-3` is strongest at
   `H20000/H30000`. The zero-sparse MLP row is now fully measured and does not
   outperform the sparse MLP control on the shorter two long horizons.
4. Project implications:
   the Dysts benchmark is no longer an execution blocker. The packet is now
   ready for benchmark-table interpretation, system-by-system comparison, and
   reuse of saved rollout artifacts for figures without further reruns.
5. Next steps:
   summarize the full `750/750` collector system by system, decide whether the
   dense-vs-blockdiag crossover should be a headline benchmark read or a
   supporting appendix read, and keep the fixed-`17` causal branch as the main
   architecture-isolation evidence.

### 2026-04-15: Extended the paper Dysts launcher and queued the missing blockdiag-MLP packet

1. Concrete results:
   the paper launcher now exposes `generic_sparse_blockdiag` as a first-class
   paper benchmark variant in
   [skae/benchmarks/paper_benchmark_manifest.py](/home/mila/l/lia/skae/skae/benchmarks/paper_benchmark_manifest.py),
   and
   [scripts/queue_paper_followup_recipes.sh](/home/mila/l/lia/skae/scripts/queue_paper_followup_recipes.sh)
   now (a) keeps comparison anchors stable when only a subset of roots is
   rerun, (b) supports `ARRAY_PARALLEL`, and (c) falls back to benchmark
   default `dt` values if the historical selected-`dt` TSV is absent. The
   first submission `9281413` failed usefully on that stale `DT_TABLE` path;
   after hardening the launcher, wrapper `9281461` completed and wrote
   [paper_followup_recipes.tsv](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415/task_tables/paper_followup_recipes.tsv)
   with `300` Dysts-only tasks (`2` roots x `15` systems x seeds `0-9`), root
   specs
   [paper_followup_recipe_roots.txt](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415/root_specs/paper_followup_recipe_roots.txt),
   and candidate roots
   [candidate_roots.csv](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415/root_specs/candidate_roots.csv).
   The live chain is training array `9281462`, collector `9281463`, and
   compare jobs `9281464-9281466`. The long-horizon Dysts reevaluation
   launcher
   [scripts/queue_dysts_long_horizon_eval.sh](/home/mila/l/lia/skae/scripts/queue_dysts_long_horizon_eval.sh)
   now accepts `INPUT_ROOT_SPECS_TSV`, and wrapper `9281481` is dependency-held
   behind `afterok:9281462` with custom root specs
   [dysts_mlp_blockdiag_long_horizon_roots.tsv](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415/root_specs/dysts_mlp_blockdiag_long_horizon_roots.tsv)
   to launch the missing `H5000/H10000/H20000/H30000` packet once training
   finishes.
2. Result in experimental context:
   the completed `750/750` Dysts long-horizon benchmark currently covers five
   roots: sparse MLP, zero-sparse MLP, dense LISTA, and two block-diagonal
   LISTA penalties. The missing architecture-side fairness control was the MLP
   with the same block-diagonal Koopman structure used by the block-diagonal
   LISTA runs. This pass closes the launcher gap rather than introducing a
   one-off Dysts-only script, which keeps the paper queue path itself as the
   source of truth for final-paper reruns.
3. Interpretation:
   there are no new forecasting results yet; the new information is execution
   state and launcher capability. The paper launcher is now broad enough to
   express all requested Dysts families, and the long-horizon reevaluation
   launcher can now target custom root sets instead of only the frozen five-root
   packet. The stale selected-`dt` path was a real launcher bug, not operator
   error, and it is now downgraded to a warning plus fallback behavior.
4. Project implications:
   if `9281462` clears, the paper will finally have a matched block-diagonal
   MLP extension on the Dysts benchmark and a queued path to measure it at the
   same long horizons as the existing five roots. That turns the current
   Dysts long-horizon benchmark from a strong five-root packet into the intended
   seven-root architecture audit without spawning another launcher family.
5. Next steps:
   watch early `9281462_*` tasks for any model-side failures, let `9281481`
   submit its reevaluation chain after the training array clears, and only then
   update the Dysts long-horizon table from the current `5`-root packet to the
   expanded `7`-root comparison.

### 2026-04-15: Fixed the blockdiag-MLP array-runner crash and requeued the Dysts packet

1. Concrete results:
   the first training array `9281462` did not produce model results; all
   tasks failed within seconds with
   `lista_linear_encoder: unbound variable` from
   [scripts/run_paper_benchmark_array.sh](/home/mila/l/lia/skae/scripts/run_paper_benchmark_array.sh)
   under `set -u`. The collector `9281463` completed but only summarized the
   fixed anchor roots already listed in
   [paper_followup_recipe_roots.txt](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415/root_specs/paper_followup_recipe_roots.txt),
   compare jobs `9281464-9281466` failed immediately, and the dependency-held
   long-horizon wrapper `9281481` was canceled when `afterok:9281462` became
   impossible. The fix is now in
   [scripts/run_paper_benchmark_array.sh](/home/mila/l/lia/skae/scripts/run_paper_benchmark_array.sh):
   optional task-table fields such as `lista_alpha`, `lista_num_loops`,
   `lista_linear_encoder`, `lista_final_op`, `decoder_coherence_weight`,
   `k_structure`, and `k_block_size` now use `${var:-}` guards. Replacement
   wrapper `9282328` completed and emitted replacement jobs:
   training array `9282330`, collector `9282331`, compare jobs `9282332-9282334`,
   and long-horizon wrapper `9282357`.
2. Result in experimental context:
   this was not a model-side instability in the new block-diagonal MLP roots.
   It was a launcher/runtime bug triggered by using the existing generic paper
   array runner with a task table that omits LISTA-only optional fields. The
   bug sat below task-table generation, so the paper launcher itself was still
   correctly expressing the Dysts packet; the crash was in the GPU execution
   layer.
3. Interpretation:
   the repaired queue is now genuinely live rather than merely submitted.
   `squeue` shows early replacement tasks `9282330_0-19` in `RUNNING` state,
   and representative task `9282330_0` is actively running on `cn-c033`
   without reproducing the original unbound-variable failure. The old `9281462`
   packet should therefore be treated as a failed first attempt, not as
   evidence against the blockdiag-MLP setup.
4. Project implications:
   the missing Dysts block-diagonal MLP fairness control is still in progress,
   but the queue is now operational. The remaining risk has shifted from
   launcher correctness to the usual training/runtime question of whether the
   packet clears cleanly on cluster resources.
5. Next steps:
   watch the early `9282330_*` tasks to confirm they write checkpoints and run
   for normal training durations, let `9282357` fire after `9282330` clears,
   and then update the Dysts long-horizon benchmark from the current `5` roots
   to the expanded `7`-root comparison once the reevaluation collector lands.

### 2026-04-16: Collected the blockdiag-MLP Dysts short-horizon packet; long-horizon wrapper never launched

1. Concrete results:
   the replacement Dysts blockdiag-MLP packet is now collected under
   [results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415).
   Replacement wrapper `9282328`, collector `9282331`, and compare jobs
   `9282332-9282334` all finished cleanly. Training array `9282330` finished
   with `299/300` successful tasks; the only failed task was `9282330_150`
   (`generic_sparse_blockdiag_ns200k_sc3em3`, `dysts:Dadras`, seed `0`), and
   its stderr shows `torch.AcceleratorError: CUDA error: uncorrectable ECC
   error encountered` on a Quadro RTX 8000 rather than a model-side crash.
   The collected short-horizon packet therefore has `149` Dysts rows for
   `sc=3e-3` and `150` for `sc=6e-3`. Dysts-only system-median best-periodic
   MSE from
   [paper_benchmark_summary.md](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415/collect/paper_benchmark_summary.md)
   and
   [forecasting_rows.csv](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415/collect/forecasting_rows.csv)
   is:
   `sc=3e-3 = 9.227e-05 / 0.001412 / 0.004684` and
   `sc=6e-3 = 7.454e-05 / 0.001399 / 0.004821` at `H100/H500/H1000`.
   System win counts are `9/6` for `sc=3e-3/sc=6e-3` at `H100`, `9/6` at
   `H500`, and `12/3` at `H1000`. The dependency-held long-horizon wrapper
   `9282357` is `CANCELLED` with no start time and wrote no files under
   [results/dysts_long_horizon_eval_mlp_blockdiag_20260415](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_mlp_blockdiag_20260415).
2. Result in experimental context:
   this closes the missing short-horizon paper-collector read for the matched
   MLP `+ block_diagonal K` controls on the `15` Dysts systems without adding a
   special-purpose launcher. The new results are directly comparable to the
   existing paper packet roots at `H100/H500/H1000`, but they do not extend the
   completed `750/750` long-horizon five-root benchmark because the custom-root
   reevaluation wrapper never launched.
3. Interpretation:
   the new blockdiag-MLP controls are competitive short-horizon Dysts models.
   On Dysts-only system medians, `sc=6e-3` is the better of the two new roots
   at `H100/H500`, while `sc=3e-3` is slightly better at `H1000`. However,
   the completed long-horizon Dysts benchmark is still the old five-root packet,
   so there is still no evidence here about whether the MLP `+ block_diagonal K`
   controls change the `H5000+` architecture ranking.
4. Project implications:
   the paper now has a real short-horizon Dysts fairness read for the
   blockdiag-MLP family, but not the matching long-horizon read. That means the
   Dysts section can discuss `H100/H500/H1000` architecture fairness with the
   new controls immediately, while any `H5000/H10000/H20000/H30000` claim still
   has to cite the completed five-root benchmark only.
5. Next steps:
   rerun only the single hardware-failed `dysts:Dadras` seed if exact
   `150/150` short-horizon coverage matters, and separately resubmit the
   canceled custom-root long-horizon wrapper so the Dysts architecture audit can
   expand from `5` roots to the intended `7` roots at `H5000/H10000/H20000/H30000`.

### 2026-04-16: Requeued the missing Dysts blockdiag-MLP long-horizon packet with a 3-hour limit

1. Concrete results:
   prior completed `H30000` reevaluation tasks from the full five-root Dysts
   packet are comfortably below `3` hours: across `750/750` completed
   `dysts_long_eval` array tasks from `9273657` and `9278881`, the elapsed-time
   summary is `min=3s`, `p50=120s`, `p95=384s`, `p99=817s`, and
   `max=1825s` (`30m25s`), with `0/750` runs above `03:00:00`. Based on that,
   [scripts/queue_dysts_long_horizon_eval.sh](/home/mila/l/lia/skae/scripts/queue_dysts_long_horizon_eval.sh)
   now accepts `EVAL_TIME_LIMIT` and uses it for both validation and full
   reevaluation array submissions. The missing short-horizon retrain is now
   queued as `9286093` with `--exclude=cn-a009` and `--time=04:00:00` for the
   single failed row `generic_sparse_blockdiag_ns200k_sc3em3`, `dysts:Dadras`,
   seed `0`. The replacement long-horizon wrapper is queued as `9286094`
   `afterok:9286093` with `EVAL_TIME_LIMIT=03:00:00`, `ARRAY_PARALLEL=48`, and
   custom root specs
   [dysts_mlp_blockdiag_long_horizon_roots.tsv](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415/root_specs/dysts_mlp_blockdiag_long_horizon_roots.tsv)
   targeting
   [results/dysts_long_horizon_eval_mlp_blockdiag_20260415](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_mlp_blockdiag_20260415).
2. Result in experimental context:
   the first long-horizon blockdiag-MLP packet failed for a trivial dependency
   reason rather than because the reevaluation path was too slow: one missing
   GPU training seed canceled the wrapper before it ran. The new queue keeps
   the same launcher and evaluation path, fixes only the missing seed, and uses
   an empirically justified shorter reevaluation limit to improve scheduler
   priority for the Dysts-long-horizon-critical work.
3. Interpretation:
   there is no need to reserve multi-day wall-clock budgets for these Dysts
   reevaluations. The completed five-root packet shows the `H30000` reevaluation
   workload is a minutes-scale CPU job, not an hours-scale or days-scale one.
   So a `03:00:00` limit is conservative and should materially improve queue
   position without risking valid `H30000` jobs timing out.
4. Project implications:
   if `9286093` clears, `9286094` should relaunch the missing blockdiag-MLP
   long-horizon packet and finally give all Dysts model families
   `H5000/H10000/H20000/H30000` coverage. That is the current paper-priority
   execution path for Dysts.
5. Next steps:
   let `9286093` finish, confirm `9286094` starts and writes its queue record,
   then report the full `H5000/H10000/H20000/H30000` system-by-system seven-root
   Dysts comparison once the collector lands.

### 2026-04-17: Completed the seven-root Dysts long-horizon architecture audit

1. Concrete results:
   retry `9286093_150` completed in `43m49s`, wrapper `9286094` completed, and
   the chained custom-root reevaluation jobs `9289755-9289758` all finished
   cleanly. The collector under
   [results/dysts_long_horizon_eval_mlp_blockdiag_20260415/collect](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_mlp_blockdiag_20260415/collect)
   now reports `300/300` complete tasks and `0` pending tasks for the two new
   blockdiag-MLP roots at `H5000/H10000/H20000/H30000`. Root-level median
   best-periodic MSE is `0.150142 / 1.14006 / 3.05356 / 3.58913` for
   `generic_sparse_blockdiag_ns200k_sc3em3` and
   `0.194482 / 1.27608 / 2.95190 / 3.47850` for
   `generic_sparse_blockdiag_ns200k_sc6em3`.
2. Result in experimental context:
   this closes the last missing long-horizon Dysts paper packet. The benchmark
   now covers all seven paper-facing Dysts roots with identical evaluation
   horizons and saved rollout artifacts:
   sparse MLP, zero-sparse MLP, dense LISTA, blockdiag LISTA `sc=3e-3`,
   blockdiag LISTA `sc=6e-3`, blockdiag MLP `sc=3e-3`, and blockdiag MLP
   `sc=6e-3`.
3. Interpretation:
   the blockdiag-MLP family is strong at the shorter long horizons but does not
   overtake the best LISTA roots at the longest ones. In aggregate, dense LISTA
   stays best at `H5000/H10000` (`0.1285/0.9778`) and blockdiag LISTA `sc=6e-3`
   stays best at `H20000/H30000` (`1.9150/2.2720`). The new blockdiag-MLP
   `sc=3e-3` root is the runner-up aggregate root at `H5000/H10000`, while the
   `sc=6e-3` root is fourth at `H20000/H30000` but still better than both plain
   MLP controls there. System-wise, the two blockdiag-MLP roots win `5/15`
   systems at `H5000` (`QiChen`, `Dadras`, `SprottTorus`, `WangSun`,
   `ShimizuMorioka`), `1/15` at `H10000` (`LorenzCoupled`), and `0/15` at
   `H20000/H30000`.
4. Project implications:
   the Dysts benchmark can now be presented as a full seven-root architecture
   audit rather than a five-root packet plus a missing-control caveat. The new
   controls sharpen the architecture story rather than overturn it: adding a
   block-diagonal Koopman operator to the MLP family materially improves over
   the plain MLP controls, especially by `H20000/H30000`, but the strongest
   long-horizon headline still favors LISTA-family models, with dense LISTA
   strongest up to `H10000` and blockdiag LISTA `sc=6e-3` strongest beyond.
5. Next steps:
   fold the seven-root system-by-system Dysts table into the senior-coauthor
   handoff, emphasize that the blockdiag-MLP controls are meaningful but not
   headline-changing at the longest horizons, and use the saved rollout caches
   from both long-horizon packets for any final phase-portrait or appendix
   figures instead of rerunning inference.

### 2026-04-17: Dysts H30000 phase-portrait packet from strongest H3000-ranked LISTA checkpoints

1. Concrete results:
   compute allocation `9295355` on `cn-m003` ran
   [tools/generate_dysts_h5000_phase_portraits.py](/home/mila/l/lia/skae/tools/generate_dysts_h5000_phase_portraits.py)
   with `--horizon 30000`, writing
   [docs/figures/dysts_phase_portraits/dysts_h30000_lista_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h30000_lista_phase_portraits_manifest.json)
   plus `15` per-system PNG/PDF/JSON packets. The H30000 rescore selected
   dense LISTA on `14/15` systems and block-diagonal LISTA (`sc=6e-3`) only
   on `dysts:Dadras`. Shared-batch H30000 best-periodic means range from
   `0.2120` on `dysts:RikitakeDynamo` to `165.3852` on `dysts:Duffing`, with
   median `1.5246`; selected periodic modes are
   `{20:2, 60:1, 80:2, 100:4, 200:4, 300:1, 400:1}`.
2. Result in experimental context:
   this reused the same H3000-shortlist then target-horizon rescore protocol
   as the H5000 and H20000 packets. It is therefore a per-system visualization
   pass over the current matched paper-facing LISTA roots, not a new training
   sweep and not the same object as the root-level aggregate benchmark table.
3. Interpretation:
   the H30000 visual packet is less block-diagonal than the H20000 packet.
   `dysts:Dadras` now becomes the lone block-diagonal selection, while
   `dysts:Duffing` and `dysts:WangSun` revert to dense LISTA under the shared-
   batch H30000 rescore. So the benchmark appendix now shows that the
   visually strongest per-system H30000 checkpoints are still mostly dense,
   even though the aggregate seven-root Dysts benchmark remains best for
   block-diagonal LISTA (`sc=6e-3`) at `H30000`.
4. Project implications:
   the senior-coauthor handoff packet can now include a fully populated
   three-horizon Dysts visual sequence (`H5000`, `H20000`, `H30000`), but the
   captioning needs to distinguish per-system visual selection from root-level
   aggregate wins. Otherwise the H30000 packet could be misread as
   contradicting the benchmark table when it is actually answering a
   different selection question.
5. Next steps:
   if we curate a small representative handoff subset, prioritize systems that
   make the horizon-dependent family mix legible, especially `dysts:Dadras`,
   `dysts:Duffing`, and `dysts:WangSun`, and cite the H30000 manifest as the
   source of truth for all selected seeds, periodic modes, and output files.

### 2026-04-17: Dysts H30000 phase-portrait packet from the true seven-root benchmark winners

1. Concrete results:
   compute allocation `9295961` on `cn-f004` ran
   [tools/generate_dysts_best_root_phase_portraits.py](/home/mila/l/lia/skae/tools/generate_dysts_best_root_phase_portraits.py),
   writing
   [docs/figures/dysts_phase_portraits/dysts_h30000_best_root_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h30000_best_root_phase_portraits_manifest.json)
   plus `15` per-system PNG/PDF/JSON packets. The packet selects the lowest
   `H30000` best-periodic forecasting MSE per system across all `7` checked-in
   Dysts long-horizon roots and all `10` seeds. Winner counts are
   block-diagonal LISTA `sc=6e-3` on `10/15` systems, block-diagonal LISTA
   `sc=3e-3` on `4/15`, and dense LISTA on `1/15` (`dysts:QiChen`). Selected
   `H30000` best-periodic means range from `0.0078` on `dysts:LuChenCheng` to
   `3.0696` on `dysts:QiChen`, with median `0.3268`; selected periodic modes
   are `{periodic_1:2, periodic_5:2, periodic_10:4, periodic_20:3,
   periodic_40:3, periodic_60:1}`.
2. Result in experimental context:
   unlike the earlier H5000/H20000/H30000 LISTA visual packets, this pass does
   not shortlist by saved `H3000` and does not rescore on a fresh shared
   batch. It reads the completed seven-root collector CSVs directly, chooses
   the per-system `H30000` winner by recorded best-periodic MSE, and renders
   from that run's stored selected-rollout artifact using the exact winning
   periodic mode.
3. Interpretation:
   the benchmark-aligned `H30000` visual story is strongly block-diagonal, not
   dense. The preferred `H30000` packet now matches the aggregate benchmark
   read far better than the older H3000-shortlist/shared-batch packet: only
   `dysts:QiChen` remains dense, while every other system selects one of the
   two block-diagonal LISTA roots.
4. Project implications:
   the senior-coauthor handoff and any appendix/slides now have a clean
   `H30000` packet that is consistent with the seven-root benchmark rather
   than potentially confusing it. This removes the need to explain the old
   dense-heavy `H30000` packet as the main visual summary; that older packet
   should now be treated as a selection-sensitivity footnote only.
5. Next steps:
   use the new best-root `H30000` manifest as the default source of truth for
   any paper-facing Dysts visual curation, and keep the older LISTA-only
   shared-batch `H30000` packet only if we explicitly want to illustrate how
   much the visual family mix depends on the selector.

## Reviewer-response mechanism branches audit and fixes (2026-04-23)

1. Concrete result:
   two evaluation-only reviewer-response branches were added, smoke-tested,
   audited, corrected, and rerun for fixed-`17` seed-`0` coverage:
   [tools/evaluate_transition_rich_true_jacobian_geometry.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_true_jacobian_geometry.py)
   with
   [scripts/run_transition_rich_true_jacobian_geometry.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_true_jacobian_geometry.sh),
   and
   [tools/evaluate_transition_rich_controlled_transfer_switching.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_controlled_transfer_switching.py)
   with
   [scripts/run_transition_rich_controlled_transfer_switching.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_controlled_transfer_switching.sh).
   Protocol documents with hypotheses, falsifiable outcomes, controls,
   metrics, and skip criteria are in
   [docs/planning/true_jacobian_geometry_experiment_20260423.md](/home/mila/l/lia/skae/docs/planning/true_jacobian_geometry_experiment_20260423.md)
   and
   [docs/planning/controlled_transfer_switching_experiment_20260423.md](/home/mila/l/lia/skae/docs/planning/controlled_transfer_switching_experiment_20260423.md).
   The first outputs are superseded because the audit found two substantive
   evaluator problems. Corrected smoke jobs `9347587` and `9347588` completed
   with exit code `0:0`. Corrected controlled-transfer shard jobs
   `9347590-9347592` completed with exit code `0:0` and wrote `1,776` rows,
   `1,632` ok rows, `144` skipped rows, and `0` failures under
   [results/controlled_transfer_switching_fixed17_seed0_20260423_corrected](/home/mila/l/lia/skae/results/controlled_transfer_switching_fixed17_seed0_20260423_corrected).
   Corrected true-geometry job `9347593` completed in `17m38s` with `49/49`
   runs, `62,460` rows, `30,014` ok rows, and `0` failures under
   [results/true_jacobian_geometry_fixed17_seed0_20260423_corrected](/home/mila/l/lia/skae/results/true_jacobian_geometry_fixed17_seed0_20260423_corrected).
2. Experimental context:
   these branches target the two reviewer vulnerabilities that the existing
   fixed-`17` evidence does not close: whether support-conditioned local laws
   agree with true local geometry near attractors, and whether support objects
   switch at measured basin-transfer events rather than merely persisting on
   autonomous within-basin trajectories.
3. Interpretation:
   the true-geometry corrected result is mixed and should be written
   cautiously. It is positive for the narrow claim that some support-family
   partitions select locally useful slopes better than count-matched random
   partitions near attractors. At radius `0.15`, blockdiag LISTA family
   relative-Frobenius error beats random for all three support definitions:
   `0.1008` vs `0.1914` for `absolute:0.001`, `0.1000` vs `0.1828` for
   `relative:0.1`, and `0.1121` vs `0.1981` for `topk:8`. Dense LISTA family
   rows also beat random across support definitions, and dense LISTA
   `topk:8` exact supports beat random at radii `0.15` and `0.3`
   (`0.1418` vs `0.2118`, `0.1296` vs `0.1721`) but not at radius `0.6`
   (`0.1456` vs `0.1453`). The result is not a clean architecture-level win:
   zero-sparse MLP rows often have lower absolute projected-Jacobian error
   because their encoder/decoder chart is closer to the state-space identity.
   Zero-sparse MLP `topk:8` exact supports are worse than random
   (`0.0847` vs `0.0463`, `0.0764` vs `0.0295`, `0.0692` vs `0.0346`
   across the radius sweep), which is a useful warning that exact support
   partitions can be arbitrary without induced sparsity.

   The controlled-transfer corrected result is sharper. Dense LISTA exact
   `topk:8` supports show the best source-to-target support switching among
   the tested roots: transfer pre-source dominance `0.8194`, post-target
   dominance `0.8230`, post-bridge target dominance `0.9370`, post-bridge lag
   `6.0455` steps, and chatter `0.0375`. The zero-sparse MLP exact
   `topk:8` row is much weaker (`0.3710`, `0.3114`, `0.3504`, lag `9.0455`),
   and blockdiag LISTA exact support collapses (`0.0172`, `0.0448`,
   `0.0519`). Support-family `topk:8` switching is nearly perfect for all
   three roots, including the zero-sparse MLP, so family-level switching is
   not sufficient evidence for a LISTA-specific sparsity mechanism.
4. Project implications:
   these results are useful but they narrow the paper claim. The true-geometry
   branch should not be used to claim recovery of true Jacobians or true
   eigendirections by LISTA supports. It can support a modest statement that
   support families sometimes select non-random local slopes in a learned
   chart, with chart-validity caveats. The controlled-transfer branch gives a
   stronger mechanistic example for dense LISTA exact `topk:8` supports, but
   the result is an explicit state-space bridge intervention, not an
   admissible optimal-control trajectory. The main paper claim should still
   rest on basin-support alignment and non-oracle self-routed forecasting.
5. Next steps:
   keep the corrected outputs as seed-`0` diagnostics and do not expand them
   until the manuscript needs one of these mechanisms in the main evidence
   chain. If expanded, prioritize seed/threshold robustness for dense LISTA
   exact `topk:8` controlled transfer and avoid framing the true-geometry
   packet as a headline result. Keep basin labels strictly evaluation-only.

### Direct periodic support-refresh ablation (April 25, 2026)

1. Concrete result:
   new evaluator and SLURM wrappers are implemented for the direct mechanism
   claim that post-entry periodic decode/re-encode refreshes latent support and
   changes later Koopman routing. The implementation files are
   [evaluate_transition_rich_periodic_support_refresh.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_periodic_support_refresh.py),
   [run_transition_rich_periodic_support_refresh.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_periodic_support_refresh.sh),
   [queue_transition_rich_periodic_support_refresh.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_periodic_support_refresh.sh),
   [merge_transition_rich_periodic_support_refresh_shards.py](/home/mila/l/lia/skae/tools/merge_transition_rich_periodic_support_refresh_shards.py),
   and
   [merge_transition_rich_periodic_support_refresh_shards.sh](/home/mila/l/lia/skae/scripts/merge_transition_rich_periodic_support_refresh_shards.sh).
   Smoke job `9361455` completed with `32` ok rows and `0` failures under
   [results/periodic_support_refresh_smoke_20260425_cal_square](/home/mila/l/lia/skae/results/periodic_support_refresh_smoke_20260425_cal_square).
   The full fixed-`17` seed-`0` LISTA-only shards completed cleanly under
   [results/periodic_support_refresh_fixed17_seed0_20260425](/home/mila/l/lia/skae/results/periodic_support_refresh_fixed17_seed0_20260425).
   Dense LISTA shard `9361464` completed `16/16` specs with `34,440` rows
   (`34,176` ok, `264` skipped, `0` failures); blockdiag LISTA shard `9361465`
   completed `17/17` specs with `38,280` rows (`38,016` ok, `264` skipped,
   `0` failures). Merge job `9361470` is still pending on scheduler priority.
   The per-root summaries give the decision-grade read:
   dense LISTA exact `topk:8` post-start refreshed-support gating has
   route-target fraction `0.8552/0.8886`, fallback `0.1392/0.1058`, and
   refreshed-versus-previous-support MSE ratio `0.0093/0.0131`. Dense LISTA
   `topk:8` family is cleaner still, with route-target fraction `1.0000/0.9996` and
   refreshed-versus-previous-support MSE ratio `0.000525/0.000626`. Blockdiag LISTA
   `topk:8` family also supports the claim (`0.9893/0.9871` route-target
   fraction). Blockdiag exact `topk:8` is weaker than dense LISTA by
   refreshed-versus-previous-support MSE and has poor target-entry behavior:
   target-entry exact-support refreshed-support routing is worse than the
   previous-support route by MSE ratio (`9.833/4.768`).
2. Experimental context:
   the older controlled-transfer branch measured whether encoder support
   objects switch along a deliberate state-space basin transfer. This new
   evaluator tests the stronger rollout mechanism by starting continuations at
   measured target-entry and post-entry states, then comparing stale source
   no-refresh, current-state no-refresh, periodic global decode/re-encode,
   previous source-support gated `K`, and refreshed-support gated `K`.
3. Interpretation:
   the result supports the stronger claim for dense LISTA `topk:8` exact
   supports once the trajectory has actually settled into the target basin
   (`post_start`), and it supports a family-level version for both LISTA roots.
   It is weaker at the first measured `target_entry`, where exact supports are
   often still transitional; dense exact `topk:8` target-entry post-reencode
   routing has fallback around `0.50`.
   It is not a positive exact-support result for blockdiag LISTA, whose exact
   post-reencode target supports remain low despite strong family behavior.
4. Project implications:
   the paper can now state the direct mechanism for the dense LISTA
   `topk:8` / post-entry setting: re-encoding from the current target-basin
   state refreshes the active support toward the target object, and
   refreshed-support gated Koopman evolution routes through target
   coordinates and strongly beats the previous-support route. The paper
   should not generalize that exact-support statement to blockdiag LISTA.
   For blockdiag, only the support-family version is currently supported.
5. Next steps:
   let merge job `9361470` complete for consolidated artifacts, then copy the
   same interpretation to `merged/periodic_support_refresh_summary.md` if the
   merged numbers match the shard summaries. No immediate new training setup
   rerun is warranted: the previously identified strongest exact-support LISTA
   setup was already the dense soft-block `p64` hard-init root, and it is the
   setup that gives the positive result here. If the manuscript needs a
   stronger robustness claim, expand dense LISTA `topk:8` post-start to more
   seeds before searching for a new root.

## Outstanding problems (active)

- **Per-basin deep-slice interpretability re-evaluation** is execution-complete:
  shard jobs `9388212`, `9388213`, `9388215`, `9388216`, and `9388217` and
  merge jobs `9388214` and `9388218` all completed with exit `0:0`. The LISTA
  packet wrote
  [9,181-line merged rows](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/interpretability_per_basin_deep_pass1/interpretability_rows.csv),
  and the boundary-MLP packet wrote
  [13,555-line merged rows](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/interpretability_per_basin_deep_pass1/interpretability_rows.csv).
  Outstanding work is now analysis/reporting, not SLURM execution: re-run or
  extend `scripts/build_per_system_stats_and_forest.py` against these new CSVs
  and add an appendix table. The existing global-slice numbers in
  `interpretability_final_pass1/` remain the main-text Table 1 source of truth.
- Documentation-to-manuscript conversion is now the immediate paper blocker:
  use the evidence order in
  [PAPER_EXPERIMENT_EVIDENCE_MAP.md](/home/mila/l/lia/skae/docs/PAPER_EXPERIMENT_EVIDENCE_MAP.md)
  to draft the actual experiments section, then backfill the main
  basin/support figure, non-oracle routing table, and Dysts long-horizon table.
  Keep queue-era chronology, tuning provenance, and superseded result packets
  out of the main narrative unless they answer a specific objection.
- **Dysts `seq_len=10`, `n=15`, H<=60K rerun is live** (queued 2026-04-28):
  Dysts training uses 30K-step source trajectories and windows of length `10`;
  held-out long-horizon evaluation will use a separate `long60` test cache to
  evaluate `H5000/H10000/H20000/H30000/H40000/H50000/H60000`. Chunk 1
  training job is `9392814`; replacement orchestrator `9393138` will submit
  the 60K eval chain after both training arrays drain. Chunk 2 is now
  `9393590`. As of 2026-04-28 15:03 EDT, the child array tasks are still
  running/pending under each array throttle, even though the orchestrator's
  compact stage log prints the top-level array state as `COMPLETED`; use
  expanded `squeue -r`/child-task state as the source of truth. Outstanding
  problem: wait for those jobs to land, collect
  `results/dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428/`, and
  redraw the Dysts table from that packet instead of the superseded 30K/100K
  plan.
- The true-Jacobian/eigendirection and controlled-transfer branches now have
  corrected seed-`0` fixed-`17` outputs. The active blocker is no longer
  execution coverage; it is claim calibration. True geometry is mixed and
  should remain a secondary falsification diagnostic because the zero-sparse
  MLP has lower absolute projected-Jacobian errors in many rows. Controlled
  transfer is paper-positive for dense LISTA exact `topk:8` support switching,
  but family-level switching is also strong for the zero-sparse MLP and the
  state-space bridge is not an admissible optimal-control transfer. The
  periodic-support-refresh packet now positively addresses the stronger claim
  for dense LISTA exact `topk:8` post-entry supports and for support-family
  routing, but not for blockdiag exact supports. The remaining blocker is
  artifact finalization from merge job `9361470` plus deciding whether the
  manuscript needs seed/threshold robustness on the narrowed dense LISTA
  exact-support claim.
- The non-oracle self-routed forecasting packet is now complete under
  [results/transition_rich_self_routed_forecasting_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420)
  with `510/510` runs complete, `24,600` rows, and `0` failures. Smoke
  validation remains under
  [results/transition_rich_self_routed_forecasting_smoke_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_smoke_20260420)
  (`270` rows, `0` failures), and the merge path remains validated under
  [results/transition_rich_self_routed_forecasting_merge_smoke_20260420/merged](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_merge_smoke_20260420/merged).
  So the active blocker is no longer missing non-oracle evidence. The paper
  now foregrounds the `H100/global` Table 2 read because it gives
  significance-backed exact-support `topk:8` routing for dense LISTA:
  `0.47 [12/17]` and `0.45 [11/17]` on the all slice, and `0.38 [14/17]`
  and `0.36 [15/17]` on the deep slice. The older `H1000/global` medians /
  win rates remain strong descriptive long-horizon context (`0.228 / 0.920`
  for `support_gated_k` and `0.275 / 0.947` for `support_local_centered`),
  but the paired Wilcoxon/Holm counts are underpowered until the seed
  expansion lands. The immediate paper task is to foreground exact `topk:8`
  routing without collapsing everything into one family-level summary.
- The centered-chart mechanism packet is now complete and resolves the earlier
  discrepancy in a materially different way than the April 19 affine-only
  diagnosis. Under
  [results/transition_rich_centered_chart_mechanism_20260420](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420),
  centered support-, family-, and basin-conditioned local laws now beat the
  learned global `K` on most deep `q4` rows across the fixed `17`, including
  the estimated-center Claude systems. So the active blocker is no longer
  “do local laws exist at all?” It is “what part of that local-law read is
  uniquely attributable to induced sparsity?” The dense no-sparsity `tanh`
  MLP now shows the same deep centered local-law effect, while the clearest
  differentiating mechanistic read is the blockdiag LISTA direct support-gated
  `K`. Boundary-adjacent `q1` states and low-coverage exact-support rows
  remain the main negative slices, so the paper has to keep the mechanism
  claim depth-qualified.
- The Dysts long-horizon execution blocker is resolved. Both long-horizon
  packets are now complete:
  [results/dysts_long_horizon_eval_20260414](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414)
  for the original five roots and
  [results/dysts_long_horizon_eval_mlp_blockdiag_20260415](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_mlp_blockdiag_20260415)
  for the two blockdiag-MLP roots. The remaining work is interpretive:
  consolidate the seven-root Dysts comparison into paper-ready tables and
  decide whether the headline should emphasize the dense-LISTA-to-blockdiag-
  LISTA crossover or instead foreground the broader induced-sparsity story,
  while explicitly separating the root-level aggregate `H30000` winner, the
  benchmark-aligned per-system best-root `H30000` packet (block-diagonal on
  `14/15` systems), and the older H3000-shortlist/shared-batch `H30000`
  packet (dense on `14/15` systems).
- The short-horizon blockdiag-MLP packet is also now complete in practice:
  retry `9286093_150` cleared the earlier hardware-transient `dysts:Dadras`
  seed-`0` gap that had come from a CUDA ECC fault on `cn-a009`, and the
  dependent long-horizon packet landed cleanly after that repair.
- The seed-`10` Dysts long-horizon benchmark packet under
  [results/dysts_long_horizon_eval_20260414](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414)
  is now complete and no longer an active queue blocker. The collector under
  [results/dysts_long_horizon_eval_20260414/collect](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/collect)
  reports `750/750` complete tasks and `0/750` pending tasks after rescue pass
  `1`. The remaining work is interpretation and paper positioning: summarize
  the full system-by-system comparison and decide how prominently to feature
  the dense-LISTA to blockdiag-LISTA crossover from `H10000` to `H20000+`.
- The supporting Dysts benchmark visual packet and the fixed-`17` LISTA
  phase-portrait packet are now complete under
  [docs/figures/dysts_phase_portraits/dysts_h5000_lista_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h5000_lista_phase_portraits_manifest.json),
  [docs/figures/dysts_phase_portraits/dysts_h20000_lista_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h20000_lista_phase_portraits_manifest.json),
  [docs/figures/dysts_phase_portraits/dysts_h30000_lista_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h30000_lista_phase_portraits_manifest.json),
  [docs/figures/dysts_phase_portraits/dysts_h30000_best_root_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h30000_best_root_phase_portraits_manifest.json),
  and
  [docs/figures/fixed17_lista_phase_portraits_20260414/fixed17_h1000_h3000_h5000_lista_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/fixed17_lista_phase_portraits_20260414/fixed17_h1000_h3000_h5000_lista_phase_portraits_manifest.json),
  so the remaining work there is narrative packaging rather than figure
  generation: choose which subset of the `15` Dysts portraits across
  `H5000/H20000/H30000` and the `51` fixed-`17` portraits belongs in the
  senior-coauthor handoff, slides, or appendix, decide whether the older or
  new `H30000` Dysts packet is shown when we only have room for one, and keep
  all of those visual packets clearly secondary to the fixed-`17` causal
  branch tables.
- The branch-wide LISTA basin-support reduction is now complete under
  [basin_support_metrics_20260408_v3](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/basin_support_metrics_20260408_v3),
  and the matched hard-init MLP fairness packet is now forecasting-complete
  under
  [results/transition_rich_hardinit_mlp_controls_seed10_20260416](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416).
  Pass-`1`
  [collect_pass1/forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/collect_pass1/forecasting_summary.md)
  and
  [dt_resolution/pass1/dt_resolution.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/dt_resolution/pass1/dt_resolution.md)
  close the rescue / `dt` side cleanly: all `51/51` arm-system pairs accepted
  the default `dt`, and the blockdiag hard-init MLP root now has system-median
  best-periodic `H100/H500/H1000 = 0.0094 / 0.0359 / 0.0383`. The active
  fairness blocker is no longer queue completion. The patched interpretability
  rerun now finishes cleanly via `9304747 -> 9304748`, writing
  [interpretability_final_pass1](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/interpretability_final_pass1)
  with `13,554` rows and `0` failures plus the matched-sampling comparison in
  [final_comparison_pass1](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1).
  That table says the two sparse hard-init MLP controls are almost tied on the
  paper slice, while the zero-sparse control remains much worse in forecasting.
  The new operator-selection mechanism study is now also complete under
  [results/transition_rich_operator_selection_hardinit_matched_20260418](/home/mila/l/lia/skae/results/transition_rich_operator_selection_hardinit_matched_20260418):
  smoke `9304650`, shards `9304655-9304659`, and merge `9304660` all finished,
  and the merged packet writes `56,538` rows with `0` failures. The important
  blocker is now scientific. Those held-out local-operator rows show that
  support families are non-random, but they do not support the strong
  `support -> local linear law` claim because even oracle basin-conditioned
  fits remain worse than one global latent law on the fixed `17` systems.
- The new study-plan state-level smoke reduction on the historical native trio
  says the next ablations must target within-basin support compression rather
  than purity alone: deep-basin `H(B|S)` is already approximately zero for
  both LISTA roots on that subset, but `H(S|B)` stays large unless support is
  forced into aggressive top-`k` masks. Exact-support uniqueness is still the
  missing positive.
- The upgraded native-trio smoke under
  [transition_rich_interpretability_smoke_20260409/native_seed0_v2_family_projection](/home/mila/l/lia/skae/results/transition_rich_interpretability_smoke_20260409/native_seed0_v2_family_projection)
  narrows that target further:
  support families already look much cleaner than exact supports, but the
  canonical exact-support counterfactual still hurts one-step prediction even
  deep inside basin while wrong-basin projections are catastrophic. The
  shortlist therefore needs either better exact-support consolidation or a
  paper story that is explicitly family/group-level rather than exact-support-
  level.
- The seed-`0` working-budget ablation ladder `v1-v7` is now complete, and
  the narrow default-sampling `200k`, `10`-seed forecast-floor packet under
  [transition_rich_basin_partition_v6_v7_200k_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_basin_partition_v6_v7_200k_seed10_20260410)
  is also complete through its fair default array / collector
  `9228394_[0-339] -> 9228395`. That forecasting-only follow-up does not
  materially improve the default-sampling floor: the better coherence root
  reaches `H100/H500/H1000 = 0.0416 / 0.0761 / 0.0796`, still behind the
  locked hard-init dense finalist and behind the matched sparse MLP control at
  `H500/H1000`, and the packet adds no new state-level reduction. The
  remaining blocker is no longer broad execution or queue completion. It is
  synthesis: lock the paper claim around the dense `p64` hard-init causal win
  over the matched sparse MLP control, and decide how strongly to foreground
  the block-diagonal forecast-retaining companion result.
- The locked `200k`, `10`-seed confirmatory packet is now fully reduced
  through rescue pass `1` under
  [transition_rich_basin_partition_final_seed10_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409).
  Pass-`1` collect / resolve `9214918 -> 9214919`, final interpretability
  reduction `9218036`, and final paired comparison `9218037` all completed
  cleanly. The main new evidence is no longer just forecasting: the packet now
  contains the matched multi-seed support analysis that the branch had been
  missing.
- The historical April 7 queue is LISTA-only, but the April 9 shortlist now
  also includes the matched sparse MLP encoder control on the same fixed
  `17` systems. That fair multi-seed paper-facing comparison is now complete.
  Its locked-packet result is asymmetric rather than uniform: the dense `p64`
  hard-init LISTA root survives the sparse MLP control plus the locked
  packet's legacy zero-`L1` ReLU control on the branch objective, while the
  block-diagonal hard-init root survives mainly as the stronger forecast-
  retaining support model.
- The hard-init seed-`0` reduction is now complete under
  [interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409/reduce/interpretability_summary.md).
  Its main result is not a clean all-roots win for near-separatrix
  oversampling. The block-diagonal hard-init variant improves deep-basin
  exact/family compression and canonical-support intervention behavior; the
  dense `p64` hard-init variant improves forecasting and canonical-support
  intervention behavior, but its raw all-state support metrics are slightly
  worse.
- The strongest current paper-facing root is now
  `lista_dense_softblock_signsplit_p64_hardinit_basin_partition` from the
  locked final packet. On the selected deep `absolute:0.001` slice it is the
  branch's clearest causal exact-support positive against the matched sparse
  MLP control (`H(S|B)=0.0543` vs `0.2449`, `U_exact=0.9923` vs `0.9772`,
  `freeze/base@20=0.1691` vs `0.3923`) while remaining forecast-competitive
  (`H1000=0.0768` vs `0.0608`). The earlier `v5`
  `lista_dense_softblock_signsplit_p64_basin_partition` root remains the
  key seed-`0` provenance result that first exposed this regime.
- The support-side basin-depth trend already agrees strongly with the intended
  narrative. In the locked packet's state-level summary at `absolute:0.001`,
  every root has much cleaner supports on the `deep` slice than on the
  `boundary` slice; for example, the dense `p64` hard-init root improves from
  `H(S|B)=3.8140` on `boundary` states to `0.8123` on `deep` states. The
  remaining missing half of that narrative is the matching state-conditioned
  forecasting read at `H100/H500/H1000`.
- The completed `v6` packet resolves the next mechanism tradeoff cleanly:
  decoder coherence helps forecasting, restrained linear pre-codes help exact-
  support compression, but neither lever nor their combination beats the
  `v5 p64` root on the joint forecast-versus-interpretability frontier.
- The new LISTA reduction rules out the stronger reusable-local-linearity
  mechanism claim on the fixed shortlist:
  local `H=20` NRMSE loses to the global fit on all `34/34` LISTA runs, and
  beats the shuffled baseline only once, on a `0.1172`-coverage dense
  `duffing_triple_well` row. The live writing task is now to separate the
  purity / reuse claim from that negative mechanism read.
- The practical step-size question is no longer open on the fixed one-seed
  LISTA shortlist. Default `dt` already clears the user-facing
  `H1000 best-periodic < 50` gate on the full `17`-system packet for both
  LISTA roots, and the overnight continuation should be read as repeated
  default-`dt` `20k` reruns caused by the former `env_dt` collection bug
  rather than as true halved-`dt` evidence.
- The next live interpretability iteration should treat
  [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md)
  as the ground-truth design inventory for the ablation axes attached to plan
  items `3` and `4`, but the branch no longer needs another broad LISTA sweep:
  momentum LISTA, adaptive / groupwise thresholds, group-aware shrinkage,
  richer reset triggers, dictionary-tied / hybrid pre-codes, and stronger
  soft-block sweeps have now all been screened through `v7`, and the
  `200k`, `10`-seed default-sampling follow-up did not promote any of them
  over the locked finalists. The remaining interpretability-study gaps
  are now primarily execution-side rather than tooling-side: wrong-support
  tests, support-switch timing diagnostics, Jacobian or operator-family
  checks, and the corresponding visual summaries are now implemented locally
  in the reducer, but they still need shortlist runs and paper-facing
  interpretation. One basis-aware caveat still remains: the completed phase-`1`
  cross-root pass uses the current reducer, which reports only the raw
  operator-family / Jacobian-family summaries. The phase-`2` alignment-aware
  readout is still a tooling gap until the reducer gains similarity-aligned
  operator distances, eigendirection similarity, and invariant-subspace /
  symmetry-aware alignment metrics. Until those are run or explicitly ruled
  out, the note is design guidance rather than evidence.
- If another LISTA-only packet is run, it should now be a narrow follow-up
  around the `v5 p64` recipe rather than another broad family sweep. The live
  default should be to compare `v5 p64`, the best forecast-retaining `v5`
  block-diagonal root, and the matched `v1` sparse MLP control first, then
  decide whether any remaining exact-support claim is still worth extra queue
  budget.
- While that diagnostic loop is still live, do not spend new interpretability
  queue budget on `200k` runs. Use `20k` diagnostics first, then rerun the
  final locked paper packet at `200k`.
- Operational queue note:
  `v4` still carries a launcher-level failure record because `9202903` hit
  `AssocMaxSubmitJobLimit`, but every task and reducer that it actually emitted
  completed successfully. There is no scheduler backlog pressure here, so any
  additional packet can be submitted deliberately rather than under queue
  triage.
- The strongest current LISTA positives within the branch are now purity /
  coverage positives rather than local-linearity positives:
  `gated_local_linear` and `gated_transfer_linear` both keep pure retained
  groups with coverage `>= 0.74`, while `multiwell_strong_transition` remains
  coverage-limited (`0.3984` block-diagonal, `0.4531` dense).
- The native-trio manual audit already says the current LISTA failure mode is
  not support contamination: support groups remain basin-pure in every
  inspected native arm. The remaining free-rollout transition issues are still
  scientifically important, but they are secondary diagnostics for this branch
  rather than its primary acceptance criterion.
- The active Claude subset is fixed to `arrested_spiral`, `cal_asymmetric_3`,
  `cal_high_cross_3`, `cal_hexagon_6`, `cal_octagon_8`, `cal_pentagon_5`,
  `cal_square_4`, `checkerboard_potential`, `duffing_triple_well`,
  `snic_multi`, `transition_routes_4`, `var_depth_gradient_4`,
  `var_diamond_4`, and `var_l_shape_5`. Broader Claude-catalog screening is
  now provenance, not an open branch-selector.
- Historical non-shortlist mechanism negatives such as Kuramoto and corrected
  competitive Lotka-Volterra should remain supporting background rather than
  live acceptance criteria for this fixed-`17` branch.

## Recent Paper-Critical Result

### April 18, 2026: pass-`1` of the matched hard-init MLP controls closed forecasting, but the interpretability reducer timed out

1. Concrete result(s):
   the fixed-`17` hard-init control packet under
   [results/transition_rich_hardinit_mlp_controls_seed10_20260416](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416)
   completed rescue pass `1` and finalized forecasting-side collection. The
   pass-`1` forecasting summary in
   [collect_pass1/forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/collect_pass1/forecasting_summary.md)
   now covers all three hard-init control roots. System-median best-periodic
   `H100/H500/H1000` is `0.0082 / 0.0260 / 0.0273` for
   `mlp_sparse_hardinit_basin_partition_control`,
   `0.0094 / 0.0359 / 0.0383` for
   `mlp_sparse_blockdiag_hardinit_basin_partition_control`, and
   `0.5704 / 2.6733 / 3.8044` for
   `mlp_zero_sparse_hardinit_basin_partition_control`. Pass-`1`
   [dt_resolution/pass1/dt_resolution.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/dt_resolution/pass1/dt_resolution.md)
   accepted the default `dt` on all `51/51` arm-system pairs and requested no
   further rescue rows. But reducer job `9295034` then timed out after
   `08:00:19`, wrote nothing under
   [interpretability_final_pass1](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/interpretability_final_pass1),
   and canceled dependent final-comparison job `9295035`.

2. Result in experimental context:
   this was the missing matched-sampling hard-init fairness packet for the
   locked `200k`, `10`-seed fixed-`17` comparison. After pass `0` showed that
   sparse hard-init MLPs could already match or beat the promoted hard-init
   LISTA forecasting winner, pass `1` was supposed to do two things: finish
   the structured blockdiag MLP forecasting table cleanly under the same
   oversampling regime, and then produce the corresponding state-level
   basin-support reduction for all three hard-init control roots.

3. Interpretation:
   the forecasting-side answer is now clear. Hard-init oversampling plus
   induced sparsity helps non-LISTA controls too, and the strongest forecasting
   root in this matched packet is still the sparse hard-init MLP rather than
   LISTA or the structured blockdiag MLP. The blockdiag hard-init MLP is still
   a strong sparse control and dramatically better than the zero-sparse tanh
   baseline, so the packet strengthens the induced-sparsity story while
   weakening any architecture-only hard-init forecasting claim. What remains
   unknown is whether the structured hard-init control improves basin-support
   alignment or intervention robustness relative to the plain sparse MLP and
   the promoted hard-init LISTA roots, because the state-level reducer did not
   finish.

4. Project implications:
   the matched-sampling hard-init forecasting table is now complete enough to
   hand to senior coauthors, and it pushes the paper narrative toward “induced
   sparsity matters, but hard-init forecasting gains are not LISTA-specific.”
   However, the corresponding interpretability table is still missing, so the
   paper cannot yet make the matched hard-init basin-support comparison that
   would separate forecasting strength from basin-support alignment strength.
   The live blocker on this branch has therefore moved from queue rescue to
   reducer scalability.

5. Next steps:
   rerun the pass-`1` hard-init interpretability reduction in a form that
   cannot lose all outputs at walltime, ideally by splitting the reducer by
   root or another bounded slice and by writing partial artifacts
   incrementally. Once those outputs exist, regenerate the canceled
   final-comparison step and update the matched hard-init architecture table
   for coauthor handoff.

### April 18, 2026: the operator-selection mechanism study was implemented, smoke-validated, and queued on the matched hard-init packet

1. Concrete result(s):
   the new offline operator-selection evaluator is now implemented in
   [tools/evaluate_transition_rich_operator_selection.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_operator_selection.py)
   with shard / merge SLURM wrappers in
   [scripts/run_transition_rich_operator_selection.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_operator_selection.sh),
   [scripts/queue_transition_rich_operator_selection_shards.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_operator_selection_shards.sh),
   and
   [tools/merge_transition_rich_operator_selection_shards.py](/home/mila/l/lia/skae/tools/merge_transition_rich_operator_selection_shards.py).
   Smoke job `9304650` completed in `00:00:29` on `cn-f002` and wrote
   [operator_selection_rows.csv](/home/mila/l/lia/skae/results/transition_rich_operator_selection_20260418/smoke_dense_seed0/operator_selection_rows.csv),
   [operator_selection_summary.md](/home/mila/l/lia/skae/results/transition_rich_operator_selection_20260418/smoke_dense_seed0/operator_selection_summary.md),
   and
   [manifest.json](/home/mila/l/lia/skae/results/transition_rich_operator_selection_20260418/smoke_dense_seed0/manifest.json)
   with `29` rows and `0` failures. The full matched-hard-init queue is now
   running under
   [results/transition_rich_operator_selection_hardinit_matched_20260418](/home/mila/l/lia/skae/results/transition_rich_operator_selection_hardinit_matched_20260418):
   shards `9304655-9304659` are running and merge `9304660` is dependency-
   queued behind them.

2. Result in experimental context:
   this is the first direct paper-facing test of whether latent supports
   actually select local linear operators rather than merely clustering
   states. For each root the queue fits held-out `A_global`, `A_basin`, and
   `A_support/family/group`, then compares them against count-matched random
   partitions, latent-kmeans controls, and masked projections of the model's
   learned global `K`. The queued packet uses the hard-init LISTA finalists
   plus the matched hard-init sparse, blockdiag-sparse, and zero-sparse MLP
   controls, with support definitions `absolute:0.001`, `relative:0.1`, and
   `topk:8` on `all`, `deep`, and `boundary` subsets.

3. Interpretation:
   the tooling side of the mechanism experiment is now closed. The smoke pass
   shows the evaluator can load the intended checkpoint families, emit rows
   and summaries, and run on a compute node without runtime failures. No
   scientific claim should be updated yet from the smoke itself; the real
   read will come from the five live shards once the fixed-`17` packet
   finishes.

4. Project implications:
   the branch now has a concrete route to answer the review's strongest
   objection on matched hard-init roots rather than only on the older
   support-purity tables. If these rows show that support or support families
   beat both one global law and matched random / geometry controls while
   masked-`K` stays competitive with the post-hoc local fits, then the paper
   can defend a mechanism claim. If not, the branch will have to soften the
   claim to basin-aligned support families without local-law selection.

5. Next steps:
   let shards `9304655-9304659` finish, merge with `9304660`, read the first
   summary table at
   [operator_selection_summary.md](/home/mila/l/lia/skae/results/transition_rich_operator_selection_hardinit_matched_20260418/operator_selection_summary.md),
   and then update the paper-track claim ladder based on whether support,
   support family, or only basin labels actually recover the best held-out
   local-law read.

### April 19, 2026: the matched hard-init interpretability rerun and the operator-selection mechanism queue both finished, and the strong local-law claim failed on the fixed-`17` packet

1. Concrete result(s):
   the patched hard-init interpretability merge reran as `9304747 -> 9304748`
   and now writes
   [interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/interpretability_final_pass1/interpretability_summary.md),
   [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1/transition_rich_final_comparison.md),
   and
   [manifest.json](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/interpretability_final_pass1/manifest.json)
   with `13,554` rows and `0` failures. On the matched hard-init control
   slice (`absolute:0.001` / `deep`), the two sparse MLP controls are nearly
   tied: blockdiag sparse MLP gives `H100/H500/H1000 = 0.0082 / 0.0252 /
   0.0264`, plain sparse MLP gives `0.0082 / 0.0260 / 0.0273`, both have
   `H(B|S) = 0.0000`, `H(S|B) = 0.2068`, `U_exact ~= 0.98`, and
   `H(F|B) = 0.0000`, while the tanh / no-shrink hard-init control remains far
   worse at `0.5704 / 2.6733 / 3.8044`.

   The operator-selection queue `9304655-9304659 -> 9304660` also completed
   cleanly and now writes
   [operator_selection_summary.md](/home/mila/l/lia/skae/results/transition_rich_operator_selection_hardinit_matched_20260418/operator_selection_summary.md)
   and
   [manifest.json](/home/mila/l/lia/skae/results/transition_rich_operator_selection_hardinit_matched_20260418/manifest.json)
   with `56,538` rows and `0` failures. Its best LISTA family rows still fail
   to beat the global operator on held-out deep transitions: dense hard-init
   LISTA reaches `partition/global = 3.8472` on `absolute:0.001` deep family
   while oracle basin fit is `31.9263`; blockdiag hard-init LISTA reaches
   `12.9785` on `topk:8` deep family while oracle basin fit is `12.3818`.
   Those family partitions are nonetheless much better than matched random or
   latent-kmeans controls: dense `3.8472` versus `365.9725` and `4.7972`,
   blockdiag `12.9785` versus `120.5169` and `24.5956`. Exact-support rows are
   frequently skipped or low-coverage (`0.1899` for blockdiag `topk:8` deep
   support, `0.6871` for dense `topk:8` deep support), and masked-`K`
   comparisons are catastrophically bad rather than supportive.

2. Result in experimental context:
   this is the review's direct mechanism test. It asks not just whether
   supports correlate with basins, but whether conditioning on support or
   support families selects a better held-out local linear operator than one
   global latent law, and whether that effect is stronger than matched random
   or geometry-matched partitions.

3. Interpretation:
   the matched hard-init control rerun strengthens the induced-sparsity story
   but weakens any architecture-only hard-init story: removing induced
   sparsity still destroys long-horizon forecasting, while the two sparse MLP
   controls remain extremely close on the deep-basin interpretability slice.
   The new operator-selection packet is a negative result for the strong
   `support -> local linear law` claim. Support families are not arbitrary
   labels because they often beat count-matched random partitions and
   sometimes latent-kmeans, but they still do not beat the global latent law.
   More importantly, even true basin partitions fail this held-out local-fit
   test, so the problem is not only support identification: on this fixed-`17`
   packet the learned latent dynamics themselves do not support the promised
   basin-specific post-hoc local linearization. The isolated boundary-slice
   anomaly in the blockdiag sparse MLP control (`relative:0.1` support
   `0.0457`) is not credible positive mechanism evidence because the same row
   has huge operator mismatch (`358.2764`) and poor masked-`K` agreement
   (`218.2634`).

4. Project implications:
   the queue blocker is closed, and the paper now has an honest answer to the
   reviewer's strongest objection. That answer supports a more modest claim:
   induced sparsity helps forecasting and yields non-random basin-aligned
   support families or dominant groups, but the current models do not show
   that support selects a valid local linear law on the fixed-`17` systems.
   If the paper keeps the full `basin -> support -> local law` storyline, the
   new mechanism packet would undercut it.

5. Next steps:
   rewrite the claim ladder and rebuttal around the weaker but defensible
   result: basin-aligned support families / dominant groups plus the value of
   induced sparsity, not exact-support local-law selection. If we still want a
   positive mechanism claim, the next experiment should not be another
   reduction of the same checkpoints; it should be a method or baseline
   change, e.g. an explicit switching/local-operator baseline or a toy-only
   state-space Jacobian / oracle-chart study.

### April 20, 2026: centered-chart mechanism packet resolves the operator-selection discrepancy and clarifies the remaining paper blocker

1. Concrete result(s):
   the reframed mechanism evaluator is now implemented in
   [tools/evaluate_transition_rich_centered_chart_mechanism.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_centered_chart_mechanism.py)
   with merge utility
   [tools/merge_transition_rich_centered_chart_mechanism_shards.py](/home/mila/l/lia/skae/tools/merge_transition_rich_centered_chart_mechanism_shards.py)
   and SLURM launchers
   [scripts/run_transition_rich_centered_chart_mechanism.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_centered_chart_mechanism.sh),
   [scripts/merge_transition_rich_centered_chart_mechanism_shards.sh](/home/mila/l/lia/skae/scripts/merge_transition_rich_centered_chart_mechanism_shards.sh),
   and
   [scripts/queue_transition_rich_centered_chart_mechanism_shards.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_centered_chart_mechanism_shards.sh).
   Smoke validation first landed cleanly under
   [results/transition_rich_centered_chart_mechanism_smoke_20260420_v2](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_smoke_20260420_v2)
   with `840` rows and `0` failures. The full SLURM chain then finished as
   `9310546-9310549` under
   [results/transition_rich_centered_chart_mechanism_20260420](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420),
   writing `74,369` rows and `0` failures
   [manifest.json](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420/manifest.json)
   [centered_chart_mechanism_summary.md](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420/centered_chart_mechanism_summary.md).

   On `relative:0.1` exact support, `persistent_current`, and deep `q4`
   states, centered support-conditioned local slopes beat the learned global
   `K` on `93.1%` of evaluated blockdiag LISTA seed-system rows (`130` rows),
   `98.6%` of dense LISTA rows (`141`), and `100%` of dense no-sparsity MLP
   rows (`140`). Deep support-gated `K` is also strongly positive:
   q4 input-gated/global-`K` win rate is `100%` for blockdiag LISTA (`121`
   rows; mean ratio `0.010`), `97.8%` for dense LISTA (`139`; mean `0.210`,
   median `0.001`), and `100%` for the dense no-sparsity MLP (`140`;
   mean `0.014`). For blockdiag LISTA, q4 block-submatrix/global-`K` is also
   `100%` wins with mean ratio `< 0.001`.

   The remaining hard slice is boundary-adjacent `q1`. Dense LISTA exact
   support wins there on only `62.5%` of evaluated rows (`24` rows), with the
   failures concentrated on `gated_local_linear` (mean
   `partition_over_global_k = 3.909`), `claude_duffing_triple_well`
   (`19.840`), and a low-coverage `claude_var_l_shape_5` row (`7.962`,
   coverage `0.085`). By contrast, deep `q4` support rows remain positive even
   on the proxy-labeled Claude systems: the `estimated_centers` subset still
   beats global `K` on `77.7%` of blockdiag LISTA rows (`130` rows), `86.2%`
   of dense LISTA rows (`130`), and `92.3%` of no-sparsity MLP rows (`130`).

2. Result in experimental context:
   this was the user-requested reframing after the affine-comparator
   representative packet showed that raw zero-intercept local fits were
   confounded by chart offsets. The new packet directly answers the correct
   mechanism question: in centered local charts, stratified by state-space
   depth and compared against the dense `tanh` / no-shrink MLP control, do
   support-, family-, or basin-conditioned local laws beat the learned global
   `K`, and does the learned `K` itself behave like a support-gated local law?

3. Interpretation:
   the earlier discrepancy was mostly methodological, not a hidden operator bug
   and not mainly a proxy-label failure. Once the local-law read is done in
   centered charts, support-conditioned local laws reappear broadly on the
   fixed `17` systems, including the estimated-center benchmark systems. The
   strong negative read from the April 18 packet was therefore mostly a
   consequence of the wrong local-fit class. The second half of the new answer
   is more cautionary: the dense no-sparsity MLP also shows strong deep
   centered local-law behavior. So the completed packet supports the existence
   of support-conditioned local laws, but it does not support the stronger
   claim that LISTA-style induced sparsity is uniquely responsible for them.

4. Project implications:
   this moves the paper out of the earlier “no local-law evidence” position.
   We can now answer the reviewer's mechanism objection with centered,
   depth-aware evidence: deep and persistent local charts do support local
   linear laws, and direct support-gated `K` is especially clean for blockdiag
   LISTA. But the paper cannot claim that this mechanism belongs only to LISTA.
   The differentiating evidence for induced sparsity still has to come from
   cleaner basin-support alignment, intervention robustness, and long-horizon
   forecasting, not merely from the existence of deep local laws.

5. Next steps:
   use this packet to build the main mechanism table and figures, then tighten
   the paper claim hierarchy. The paper should now say that centered local-law
   evidence exists, that it is strongest away from separatrices, and that
   blockdiag LISTA gives the cleanest direct support-gated `K` interpretation.
   The follow-up work should focus on tying that local-law read back to
   support uniqueness and forecasting by depth, not on rerunning more
   zero-intercept local-fit studies.

### April 20, 2026: non-oracle self-routed forecasting packet is implemented, smoke-validated, and launch-ready

1. Concrete result(s):
   the deployment-facing self-routed forecasting evaluator is now implemented
   in
   [tools/evaluate_transition_rich_self_routed_forecasting.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_self_routed_forecasting.py)
   with merge utility
   [tools/merge_transition_rich_self_routed_forecasting_shards.py](/home/mila/l/lia/skae/tools/merge_transition_rich_self_routed_forecasting_shards.py)
   and SLURM launchers
   [scripts/run_transition_rich_self_routed_forecasting.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_self_routed_forecasting.sh),
   [scripts/merge_transition_rich_self_routed_forecasting_shards.sh](/home/mila/l/lia/skae/scripts/merge_transition_rich_self_routed_forecasting_shards.sh),
   and
   [scripts/queue_transition_rich_self_routed_forecasting_shards.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_self_routed_forecasting_shards.sh).

   Single-spec smoke validation first landed cleanly under
   [results/transition_rich_self_routed_forecasting_smoke_20260420_single](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_smoke_20260420_single),
   where dense hard-init LISTA on `gated_transfer_linear`, seed `0`, wrote
   `8` rows with `0` failures and completed in `0.6s`. The reduced
   three-system / three-root smoke then landed under
   [results/transition_rich_self_routed_forecasting_smoke_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_smoke_20260420),
   writing `270` rows (`271` CSV lines including header) with `0` failures
   across `9` reduced runs
   [manifest.json](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_smoke_20260420/manifest.json)
   [self_routed_forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_smoke_20260420/self_routed_forecasting_summary.md).
   Merge-path validation is also complete under
   [results/transition_rich_self_routed_forecasting_merge_smoke_20260420/merged](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_merge_smoke_20260420/merged),
   after fixing a summary-merge bug caused by CSV string values in the summary
   reducer.

2. Result in experimental context:
   this is the user-requested follow-up to the centered-chart mechanism study.
   The centered packet answered the chart-level question, but it still
   allowed offline operator analysis. This new packet answers the actual
   deployment-facing forecasting question: can the model's own current support
   or support family route a rollout into a better local law than one global
   `K`, without oracle basin labels at forecast time?

   The implemented rollout modes are:
   `global_k`, `support_gated_k`, `support_block_gated_k`,
   `support_local_centered`, and `family_local_centered`. The packet fits
   support- and family-conditioned local centered operators on separate fit
   trajectories, then evaluates held-out rollout starts stratified by initial
   depth (`all`, `q1`, `q4` in the smoke; `all,q1,q2,q3,q4` in the full
   default). It also records route coverage, fallback fraction, switch rate,
   and per-horizon `H / global` ratios.

3. Interpretation:
   the technical blocker is now closed. We no longer just have a proposal for
   the non-oracle read; we have working evaluator, merge path, and shard
   queueing scripts on the same checkpoint sources used by the centered-chart
   packet. The smoke summary also confirms that the packet is numerically
   doing the right kind of work: it produces finite non-oracle family-routed
   and support-gated rollout rows, records route coverage/fallback numbers,
   and preserves the depth-sliced reporting we need for the paper.

4. Project implications:
   the next paper-critical queue is now sharply defined. The key unresolved
   forecasting claim is no longer “how should we test non-oracle local-law
   routing?” It is “what does the full fixed-`17` packet say once we test it?”
   This packet is the direct bridge from the centered-chart mechanism result
   to the paper's deployment-facing claim about induced sparsity, basin-aware
   routing, and long-horizon forecasting.

5. Next steps:
   submit the full fixed-`17` self-routed forecasting packet over the same
   three roots used in the centered-chart study, sharded by root, then compare
   `global_k` against `support_gated_k`, `support_block_gated_k`,
   `support_local_centered`, and `family_local_centered` at `H100/H500/H1000`
   overall and by initial-state depth. The paper-facing read should emphasize
   support family as the likely deployment router and exact support as a
   stricter mechanistic analysis, not the only routing object.

### April 20, 2026: full fixed-`17` self-routed forecasting packet completed, and dense LISTA exact-support routing is the strongest non-oracle result

1. Concrete result(s):
   the full non-oracle self-routed forecasting packet is now complete under
   [results/transition_rich_self_routed_forecasting_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420)
   with automation manifest
   [self_routed_forecasting_queue.json](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420/automation/self_routed_forecasting_queue.json).
   Shards `9314443-9314472` and merge `9314473` all finished `COMPLETED 0:0`,
   with shard elapsed times ranging from `00:39:51` to `02:08:48`. The merged
   packet writes
   [self_routed_forecasting_rows.csv](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420/self_routed_forecasting_rows.csv),
   [self_routed_forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420/self_routed_forecasting_summary.md),
   and
   [manifest.json](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420/manifest.json)
   with `510/510` runs complete, `24,600` rows, and `0` failures.

   On the paper-facing non-oracle `H1000/global` read, the strongest exact-
   support router is dense LISTA with `topk:8`. All-slice median ratios / win
   rates are `0.228 / 0.920` for `support_gated_k` and `0.275 / 0.947` for
   `support_local_centered`, with median coverage about `0.53`. Deep `q4`
   states are slightly stronger at `0.224 / 0.923` and `0.207 / 0.985`.
   Blockdiag LISTA is also positive but lower-coverage:
   `0.832 / 0.739` and `0.801 / 0.783` all-slice, with median coverage about
   `0.12`; the direct `support_block_gated_k` read stays interpretable but is
   weaker at `0.983 / 0.696` all-slice and `0.958 / 0.684` on deep `q4`, with
   only about `0.003` median coverage. The dense no-sparsity `tanh` MLP is
   much weaker on the same exact-support router:
   `0.924 / 0.539` for `support_gated_k` and `1.000 / 0.496` for
   `support_local_centered` all-slice; deep `q4` remains only
   `0.964 / 0.519` and `1.000 / 0.473`.

2. Result in experimental context:
   this is the deployment-facing follow-up to the centered-chart mechanism
   study. It asks whether the model's own inferred support or support family,
   without oracle basin labels, can actually route forecasting better than one
   global Koopman matrix. The packet covers the three centered-chart roots on
   the full fixed `17`, with support definitions `relative:0.1` and `topk:8`,
   rollout modes `global_k`, `support_gated_k`, `support_block_gated_k`,
   `support_local_centered`, and `family_local_centered`, and depth strata
   `all`, `q1`, `q2`, `q3`, and `q4`.

3. Interpretation:
   the core reviewer-facing question now has a positive, non-oracle answer,
   but only for the right routing object. Exact-support `topk:8` routing is
   the clean deployment result: it consistently beats one global `K` on the
   dense LISTA root and beats it more modestly on the blockdiag LISTA root.
   The zero-sparsity `tanh` MLP is much weaker on the same self-routed exact-
   support evaluation, which is the cleanest current evidence that induced
   sparsity helps produce a usable routing signal rather than merely local
   laws in hindsight.

   At the same time, not every support object works. Thresholded exact support
   (`relative:0.1`) is too fragmented for deployment: on deep `q4`,
   `support_gated_k` is skipped `160/160` times on each root with
   `support_class_count>max_partition_classes`. Family routing is higher
   coverage and often excellent on LISTA medians (`H1000/global` all-slice
   medians `2.9e-4` blockdiag and `2.2e-3` dense), but the mean tables explode
   because a minority of catastrophic rollouts dominate. That instability is
   especially severe on the zero-sparsity MLP, where family routing is simply
   bad (`H1000/global` median `54.6` all-slice and `47.6` on deep `q4`).

4. Project implications:
   this packet lets the paper finally connect the interpretability story to a
   real forecasting mechanism without oracle basin labels. The strongest safe
   claim is now: dense LISTA exact-support `topk:8` routing often selects a
   better local forecasting law than one global `K`, and it does so much more
   reliably than the dense zero-sparsity control. That is a stronger and more
   directly deployable result than the earlier centered one-step analysis.
   The paper should not lead with `relative:0.1` exact supports or with family
   routing as the main deployment object.

5. Next steps:
   build the paper table around exact-support `topk:8` self-routing at
   `H100/H500/H1000`, stratified by depth (`q1` versus `q4`), with dense LISTA
   as the main positive and the zero-sparsity MLP as the decisive control.
   Keep family routing as a secondary table or appendix because its median
   behavior is strong on LISTA but its catastrophic tail will otherwise muddy
   the main message. If needed, add one short tail-diagnostic figure showing
   that family routing is high-coverage but unstable, while exact `topk:8`
   routing is lower-coverage but cleaner.

### April 19, 2026: support-flow, proxy-label, and affine-comparator diagnostics explain the toy-versus-benchmark discrepancy

1. Concrete result(s):
   the new support-flow diagnostic is now implemented in
   [tools/diagnose_transition_rich_support_flow.py](/home/mila/l/lia/skae/tools/diagnose_transition_rich_support_flow.py)
   with SLURM launcher
   [scripts/run_transition_rich_support_flow.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_support_flow.sh).
   Smoke packets landed cleanly under
   [results/transition_rich_support_flow_smoke_20260419/dense_gated_local_cal_square_seed0](/home/mila/l/lia/skae/results/transition_rich_support_flow_smoke_20260419/dense_gated_local_cal_square_seed0)
   and
   [results/transition_rich_support_flow_smoke_20260419/gated_transfer_dense_seed0](/home/mila/l/lia/skae/results/transition_rich_support_flow_smoke_20260419/gated_transfer_dense_seed0),
   both with `0` failures. On deep `relative:0.1` exact support, the learned
   dense LISTA global `K` already keeps almost all one-step latent energy on
   the current support: `full_output_on_mask_energy_fraction = 0.9953` on
   `gated_local_linear`, `0.9959` on `gated_transfer_linear`, and `0.9927` on
   `claude:cal_square_4`
   [support_flow_summary.md](/home/mila/l/lia/skae/results/transition_rich_support_flow_smoke_20260419/dense_gated_local_cal_square_seed0/support_flow_summary.md)
   [support_flow_summary.md](/home/mila/l/lia/skae/results/transition_rich_support_flow_smoke_20260419/gated_transfer_dense_seed0/support_flow_summary.md).

   The same smoke packet also shows that proxy labels are not the main deep
   failure mode on the native toys. `gated_local_linear` has exact native
   agreement for both `env_points` and `estimated_centers` on deep and
   boundary states (`1.0000` matched accuracy, `1.0000` NMI), while
   `gated_transfer_linear` is also exact on deep states and only modestly
   noisy on boundary states (`0.9307` for `env_points`, `0.9491` for
   `estimated_centers`)
   [label_agreement_summary.md](/home/mila/l/lia/skae/results/transition_rich_support_flow_smoke_20260419/dense_gated_local_cal_square_seed0/label_agreement_summary.md)
   [label_agreement_summary.md](/home/mila/l/lia/skae/results/transition_rich_support_flow_smoke_20260419/gated_transfer_dense_seed0/label_agreement_summary.md).

   The decisive diagnostic is the compact representative affine packet under
   [results/transition_rich_operator_selection_affine_representatives_direct_20260419](/home/mila/l/lia/skae/results/transition_rich_operator_selection_affine_representatives_direct_20260419).
   It finished in the active compute allocation with `40/40` runs complete,
   `1,159` rows, and `0` failures
   [manifest.json](/home/mila/l/lia/skae/results/transition_rich_operator_selection_affine_representatives_direct_20260419/manifest.json).
   The saved comparison note
   [linear_vs_affine_representative_comparison.md](/home/mila/l/lia/skae/results/transition_rich_operator_selection_affine_representatives_direct_20260419/linear_vs_affine_representative_comparison.md)
   shows that the strongest deep toy positives mostly collapse once the
   comparator is upgraded from zero-intercept linear to affine. On deep
   `relative:0.1`, `gated_local_linear` basin/family goes from `0.3869` to
   `3.2650` and support from `0.4034` to `3.4420`; `gated_transfer_linear`
   basin/family goes from `0.6386` to `0.9223` and support from `0.7861` to
   `1.1190`.

2. Result in experimental context:
   this was the targeted discrepancy diagnosis after the fixed-`17`
   operator-selection table came back negative overall but still showed
   positive toy-system rows. The immediate questions were whether the toy
   positives were artifacts of bad proxy labels, whether support was failing
   to activate the learned global `K`, or whether the evaluation itself was
   rewarding the wrong class of local fit.

3. Interpretation:
   the evidence now points primarily to a fit-class mismatch rather than a
   support-activation bug. Deep proxy labels on the native toys are already
   exact, and the learned global `K` already keeps nearly all one-step latent
   energy on the current exact support for both the native toys and a
   representative Claude system. So the discrepancy is not well explained by
   “support does not activate the operator” or by “proxy labels are wrong
   deep inside a basin.” The representative affine packet instead shows that
   the earlier deep toy local-law win was largely coming from separate
   zero-intercept local fits absorbing chart offsets / affine terms that one
   global zero-intercept fit could not absorb. Once the global comparator is
   allowed to be affine, most of that apparent deep local-law advantage
   disappears.

4. Project implications:
   this resolves the main methodological ambiguity behind the reviewer's
   strongest objection. We do not need to hypothesize a hidden implementation
   bug to explain why toy operator-selection looked cleaner than the broader
   packet. The cleaner explanation is that the earlier mechanism read was
   substantially confounded by the class of post-hoc local fit. The current
   branch therefore supports basin-support alignment and induced sparsity more
   than support-selected local linear laws.

5. Next steps:
   if we keep the current checkpoints, write the paper around induced sparsity
   plus basin-aligned supports / support families and explicitly say the
   strong local-law claim was not robust to an affine comparator. If we still
   want a positive mechanism claim, the next experiment should be an explicit
   local-affine or switching baseline, or a centered / chart-normalized local
   operator study, not another zero-intercept reduction of the same models.

### April 17, 2026: pass-`0` of the matched hard-init MLP controls landed, and the blockdiag control is still in rescue

1. Concrete result(s):
   the fixed-`17` matched hard-init control packet under
   [results/transition_rich_hardinit_mlp_controls_seed10_20260416](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416)
   has finished launcher `9285895`, initial training array `9285897`, and
   pass-`0` collect / resolve / advance `9285898 -> 9285899 -> 9285900`.
   Array `9285897` ended with `495` `COMPLETED` and `15` `FAILED` tasks, with
   failures split as `3` sparse-control rows, `3` zero-sparse rows, and `9`
   blockdiag-control rows. The pass-`0` collector wrote accepted-default
   forecasting summaries for the two plain-encoder controls in
   [collect_pass0/forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/collect_pass0/forecasting_summary.md):
   `mlp_sparse_hardinit_basin_partition_control` reaches system-median
   best-periodic `H100/H500/H1000 = 0.0082 / 0.0260 / 0.0273` on `167/170`
   rows, while `mlp_zero_sparse_hardinit_basin_partition_control` reaches
   `0.5704 / 2.6733 / 3.8044` on `167/170` rows. Pass-`0` dt resolution sent
   the structured control to rescue on all `17` systems, and as of
   `2026-04-17 09:57 EDT` rescue array `9291399` is `154` `COMPLETED`,
   `15` `FAILED`, `1` `RUNNING`, with dependent collect / resolve / advance
   `9291400 -> 9291401 -> 9291402` still dependency-held.

2. Result in experimental context:
   this was the missing matched-sampling hard-init fairness packet for the
   locked `200k`, `10`-seed fixed-`17` comparison. Its job was to answer the
   clean question that the earlier mixed-regime packet could not answer:
   whether the strong hard-init forecasting read survives once the MLP
   controls are trained with the same near-separatrix oversampling scheme as
   the promoted LISTA roots.

3. Interpretation:
   the answer is already asymmetric on pass `0`. Hard-init oversampling does
   transfer to a non-LISTA sparse model: the sparse hard-init MLP control is
   not just competitive, it is provisionally better on `H1000` than both the
   locked standard-sampling sparse MLP control (`0.0608`) and the locked
   hard-init LISTA forecasting winner (`0.0516`). So the old forecasting gain
   cannot be claimed as LISTA-specific. At the same time, the tanh / no-shrink
   hard-init control is dramatically worse, so induced sparsity still looks
   essential. The remaining unknown is the structured blockdiag hard-init MLP
   read, not whether hard-init helps sparse MLPs at all. Every rescue failure
   inspected so far is the same CUDA ECC crash on `cn-a009`, so there is still
   no evidence of a model-side bug.

4. Project implications:
   the paper story needs to move away from any architecture-only forecasting
   claim under hard-init. The strongest forecasting implication from the new
   pass is instead: near-separatrix hard-init oversampling can help sparse
   models broadly, while zero-sparsity controls remain much weaker. That makes
   the main paper distinction under hard-init induced sparsity versus no
   induced sparsity, with LISTA's remaining value more likely to live in
   support structure and interpretability than in a unique forecasting edge.

5. Next steps:
   let rescue pass `1` finish or freeze the remaining hardware-limited rows,
   collect the structured blockdiag hard-init MLP summary, and then update the
   fixed-`17` comparison tables so standard-sampling and hard-init rows are
   explicitly separated and the new sparse-hard-init MLP result is treated as
   a real headline constraint on the paper narrative.

### April 11, 2026: the default-sampling `200k`, `10`-seed `v6` / `v7` forecast-floor check completed and did not reopen the shortlist

- Concrete result(s):
  the narrow long-budget default-sampling follow-up under
  [transition_rich_basin_partition_v6_v7_200k_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_basin_partition_v6_v7_200k_seed10_20260410)
  completed its fair comparison as default array / collector
  `9228394_[0-339] -> 9228395` on April 11, 2026. The collector wrote `329`
  rows from `2` roots across all `17` systems with no collector stderr
  ([collect-transition-rich-9228395.out](/network/scratch/l/lia/skae/collect-transition-rich-9228395.out)).
  The wrapper `9228393` shows `FAILED` only because its attempted model-wise
  dt-rescue continuation hit `AssocMaxSubmitJobLimit` after the default table
  had already been emitted
  ([queue-transition-rich-dt-9228393.err](/network/scratch/l/lia/skae/queue-transition-rich-dt-9228393.err)).
  The better of the two roots,
  `lista_dense_softblock_signsplit_coherence_basin_partition`, reaches
  system-median best-periodic `H100/H500/H1000 = 0.0416 / 0.0761 / 0.0796`,
  while `lista_blockdiag_sparsegroup_basin_partition` lands at
  `0.0437 / 0.1142 / 0.1193`. Both roots still cover all `17` systems, though
  the collector is missing `6` and `5` seed-system rows respectively.

- Result in experimental context:
  this was the only remaining long-budget default-sampling LISTA follow-up
  after the locked hard-init confirmatory packet. Its role was deliberately
  narrow: test whether the best forecasting roots from `v6` and `v7` could
  lower the default-sampling forecast floor enough to justify reopening the
  promoted lineage before coauthor handoff.

- Interpretation:
  the answer is no. The stronger coherence root is still slightly worse than
  the promoted dense hard-init finalist at all three horizons
  (`0.0196 / 0.0733 / 0.0775`) and worse than the matched sparse MLP control
  on the long horizons that matter most (`0.0614 / 0.0608` at `H500/H1000`).
  The sparsegroup root is farther behind. Because this packet is
  forecasting-only and never ran a state-level interpretability reduction, it
  adds no new basin-support positive that could offset that forecast result.

- Project implications:
  the basin-support interpretability branch is now training-closed. The locked
  hard-init packet remains the only paper-facing state-level basin-support
  win, the block-diagonal hard-init root remains the forecast-retaining
  companion, and the default-sampling `v6` / `v7` roots remain appendix /
  shortlist-provenance evidence rather than promoted finalists.

- Next steps:
  do not reopen method search. Spend remaining branch effort on the
  state-conditioned `H100/H500/H1000` read, support-family versus exact-
  support claim calibration, and any matched-sampling MLP controls needed to
  make the final fairness language defensible.

### April 10, 2026: the clean locked-budget tanh / no-shrink control and the narrow default-sampling LISTA refinement both completed

- Concrete result(s):
  the older array
  [transition_rich_zero_sparse_control_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_zero_sparse_control_seed10_20260410)
  was confirmed to be misconfigured: its generated task table labels
  `mlp_zero_sparse_basin_partition_control` but sets
  `config_name=generic_sparse` rather than `generic_no_shrink`. To correct
  that, a fresh locked-budget packet was submitted as wrapper job `9224111`
  under
  [transition_rich_zero_sparse_tanh_control_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_zero_sparse_tanh_control_seed10_20260410)
  with `MODEL_VARIANTS_CSV=mlp_zero_sparse_basin_partition_control`,
  `SEEDS_CSV=0,1,2,3,4,5,6,7,8,9`, `NUM_STEPS_OVERRIDE=200000`, and
  `MAX_HALVINGS=0` so the default array is queued cleanly before any rescue
  passes are considered. That wrapper now completes cleanly as
  `9224263_* -> 9224264 -> 9224265`: the task table is rebuilt with
  `config_name=generic_no_shrink`, the collector writes `169` rows despite one
  fast array failure, all `17/17` systems accept default `dt`, and the clean
  no-shrink control reports system-median best-periodic
  `H100/H500/H1000 = 0.5763 / 1.7924 / 2.4279`. The older ReLU-only ablation
  packet also now runs through collect / resolve pass `4` with no smaller-`dt`
  requests and remains at
  `H100/H500/H1000 = 0.5764 / 2.0556 / 2.6532`.
  The narrow default-sampling LISTA follow-up under
  [transition_rich_basin_partition_20260410_seed0_smoke_v7](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260410_seed0_smoke_v7)
  also finished as `9226564_[0-135] -> 9226565 -> 9226566 -> 9226567` with
  `0` reducer failures and one fast training failure on
  `lista_dense_softblock_signsplit_p64_softblock5em4_basin_partition` for
  `gated_transfer_linear`. Among the new roots, the best forecasting read is
  `lista_blockdiag_sparsegroup_basin_partition`
  (`H1000 system-median best = 0.0846`), followed by
  `lista_dense_softblock_hybrid_precode_basin_partition`
  (`0.0959`). The strongest new deep-basin support-compression read is
  `lista_dense_softblock_signsplit_p64_softblock5em4_basin_partition` with
  `H(S|B)=0.6795`, `U_exact=0.8453`, and `H(F|B)=0.0634` at
  `absolute:0.001` / `deep`.

- Result in experimental context:
  the clean no-shrink packet closes the last missing training-side control on
  the fixed `17` systems: the branch now has the locked LISTA-vs-sparse-MLP
  read, the ReLU-only zero-`L1` ablation, and the decisive full-budget,
  `10`-seed tanh / no-shrink MLP control. The `v7` packet was the last cheap
  method-side default-sampling screen around the current winners before
  deciding whether any additional recipe search was justified.

- Interpretation:
  the stale `9221521_*` packet should still not be used as the clean no-shrink
  anti-sparsity control, because it retains ReLU-induced shrinkage. But the
  pair of completed controls now sharpens the same conclusion from two angles:
  removing only the explicit `L1` penalty already leaves the branch much weaker
  than the promoted sparse LISTA roots, and removing induced sparsity more
  completely with the tanh / no-shrink control still does not recover that
  gap. The `v7` packet does not produce a single new recipe that is both
  forecast-leading and support-leading: `sparsegroup` forecasts best among the
  new roots but has weaker support alignment, while `softblock5e-4` gives the
  strongest new support-compression read but is incomplete on one system and
  does not become the clean new forecasting leader. Adaptive/groupwise
  thresholds are unstable. The training-side claim therefore remains broader
  than “LISTA wins because of one exact architecture”: some induced sparsity is
  necessary for good finite-dimensional Koopman representations in multi-basin
  systems, and LISTA is one structured way to realize that sparsity.

- Project implications:
  the training-side evidence is now effectively closed for paper positioning.
  The clean no-shrink contrast can be cited directly as the decisive
  anti-sparsity control, while the `v7` packet should be treated as shortlist
  provenance rather than as a reason to reopen the promoted finalists. We do
  not need another broad architecture sweep before coauthor handoff.

- Next steps:
  run the state-conditioned `H100`, `H500`, and `H1000` analysis on the locked
  finalists and controls by basin depth / separatrix proximity. If any further
  method-side work is done at all, it should be a very narrow tie-break between
  `sparsegroup` and `softblock5e-4`, not another broad LISTA sweep.

### April 10, 2026: the locked `10`-seed confirmatory packet finalized through pass `1` and the dense `p64` hard-init root is the only clear causal basin-support winner

- Concrete result(s):
  the locked packet
  [transition_rich_basin_partition_final_seed10_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409)
  finished rescue pass `1` collection / resolution as `9214918 -> 9214919`,
  then completed the final state-level reducer `9218036` and paired summary
  `9218037`. The packet is finalized with `0` remaining request rows in
  [advance_pass1.json](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/automation/advance_pass1.json)
  and `0` interpretability failures in
  [failures.json](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/interpretability_final_pass1/failures.json).
  The finalized forecasting summary under
  [collect_pass1/forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/collect_pass1/forecasting_summary.md)
  reports system-median best-periodic `H1000` values of `0.0516` for the
  block-diagonal hard-init LISTA root, `0.0775` for the dense soft-block
  `p=64` hard-init LISTA root, `0.0608` for the matched sparse MLP control,
  and `0.0909` for the zero-`L1` ReLU MLP ablation. On the selected
  `absolute:0.001` / `deep` slice in
  [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md),
  the dense `p64` hard-init root reaches `H(S|B)=0.0543`, `U_exact=0.9923`,
  and `freeze/base@20=0.1691` against the matched sparse MLP control's
  `0.2449`, `0.9772`, and `0.3923`, with paired wins on `15/17`, `14/17`,
  and `16/17` systems.

- Result in experimental context:
  this is the locked `200k`, `10`-seed, fixed-`17` paper packet that was
  explicitly reserved for the final causal comparison between the two promoted
  LISTA finalists and the sparse MLP control, with an additional legacy
  zero-`L1` ReLU control. It is the first branch result that combines the full
  multi-seed forecasting read with the same study-plan state-level
  interpretability reducer and the paired LISTA-vs-control summary on one
  fixed selected slice.

- Interpretation:
  the multi-seed result does not support a uniform “LISTA always wins”
  headline. It supports a narrower but cleaner one. The dense soft-block
  `p=64` hard-init LISTA root is the branch's only clear basin-support winner
  over the matched sparse MLP control on the selected slice. The
  block-diagonal hard-init root remains valuable, but for a different reason:
  it is the strongest forecast-retaining root and it improves wrong-support ablation
  robustness relative to both MLP controls, yet it loses exact-support
  compression to the matched sparse MLP control. Also, `H(F|B)` is `0.0000`
  for all four roots on this slice, so family entropy no longer separates the
  candidates here; the real discriminators are exact-support fragmentation,
  intervention stability, persistence, and forecasting.

- Project implications:
  the main paper-facing LISTA-vs-sparse-MLP question is answered at the locked
  budget. The strongest defensible main-text claim from this packet is no
  longer “some LISTA root beats MLP in general.” It is that the dense `p64`
  hard-init LISTA recipe produces markedly cleaner deep-basin exact-support
  alignment than the matched sparse MLP control while remaining forecast-
  competitive on the fixed `17` systems. The block-diagonal hard-init root
  should be positioned as the supporting forecast-retaining companion, not as
  the lead exact-support result. The broader induced-sparsity claim still
  depends on the clean tanh / no-shrink control now queued separately.

- Next steps:
  rewrite the coauthor-facing branch summary around the dense `p64` causal win,
  update the queue status and paper-track claim accordingly, and use the next
  remaining effort on paper-facing visuals plus any basis-aware alignment
  diagnostics needed to decide whether the final wording stays at
  exact-support reuse or softens to family / dominant-group or
  symmetry-aware-alignment language.

- Fixed-`17` forecasting snapshot for the locked roots:
  the table below records finalized best-periodic system medians from
  [collect_pass1/forecasting_rows.json](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/collect_pass1/forecasting_rows.json),
  reported as `H100 / H500 / H1000`. It is the compact per-system forecasting
  view for the four best branch setups currently documented: the two promoted
  LISTA finalists plus the sparse MLP control and the zero-`L1` ReLU
  MLP control from the locked packet.

| system | blockdiag hard-init LISTA | dense `p64` hard-init LISTA | sparse MLP control | zero-`L1` ReLU MLP ablation | best `H1000` root |
|---|---:|---:|---:|---:|---|
| multiwell_strong_transition | `5.67e+03 / 1.07e+27 / 1.0421` | `3.38e+03 / 1.16e+27 / 0.0793` | `157.9347 / 1.42e+20 / 1.0787` | `187.1979 / 1.12e+19 / 0.5794` | dense `p64` hard-init LISTA |
| gated_local_linear | `0.0010 / 0.000788 / 0.000795` | `0.0011 / 0.0014 / 0.0014` | `0.0017 / 0.0015 / 0.0014` | `0.0290 / 0.0427 / 0.0443` | blockdiag hard-init LISTA |
| gated_transfer_linear | `0.2618 / 0.7842 / 0.8188` | `0.2675 / 0.7964 / 0.8612` | `0.1974 / 0.6995 / 0.7704` | `0.1908 / 0.7296 / 0.8037` | sparse MLP control |
| arrested_spiral | `0.0095 / 0.0126 / 0.0131` | `0.0292 / 0.0355 / 0.0364` | `0.0153 / 0.0245 / 0.0257` | `0.0203 / 0.0280 / 0.0290` | blockdiag hard-init LISTA |
| cal_asymmetric_3 | `0.0206 / 0.0707 / 0.0770` | `0.0360 / 0.0804 / 0.0864` | `0.0046 / 0.0338 / 0.0377` | `0.0257 / 0.0714 / 0.0771` | sparse MLP control |
| cal_high_cross_3 | `0.1338 / 0.3034 / 0.3300` | `0.1502 / 0.2792 / 0.2956` | `0.3688 / 0.8289 / 0.8885` | `0.1792 / 0.4364 / 0.4591` | dense `p64` hard-init LISTA |
| cal_hexagon_6 | `0.0283 / 0.0491 / 0.0516` | `0.0196 / 0.0267 / 0.0279` | `0.0271 / 0.0481 / 0.0507` | `0.0434 / 0.0725 / 0.0761` | dense `p64` hard-init LISTA |
| cal_octagon_8 | `0.0384 / 0.0523 / 0.0542` | `0.0565 / 0.0876 / 0.0916` | `0.1304 / 0.2251 / 0.2324` | `0.1290 / 0.1922 / 0.2020` | blockdiag hard-init LISTA |
| cal_pentagon_5 | `0.0226 / 0.0376 / 0.0395` | `0.0399 / 0.0733 / 0.0775` | `0.0297 / 0.0703 / 0.0751` | `0.0604 / 0.1080 / 0.1138` | blockdiag hard-init LISTA |
| cal_square_4 | `0.0025 / 0.0020 / 0.0019` | `0.0117 / 0.0112 / 0.0112` | `0.0394 / 0.0580 / 0.0604` | `0.0406 / 0.0584 / 0.0605` | blockdiag hard-init LISTA |
| checkerboard_potential | `2.0022 / 2.69e+03 / 0.0776` | `0.1605 / 0.1526 / 0.1086` | `0.1641 / 0.0614 / 0.0608` | `0.0924 / 0.1366 / 0.0909` | sparse MLP control |
| duffing_triple_well | `0.0041 / 0.0054 / 0.0039` | `0.0036 / 0.0065 / 0.0038` | `0.000958 / 0.0017 / 0.0011` | `0.0012 / 0.0026 / 0.0014` | sparse MLP control |
| snic_multi | `0.000197 / 0.0003 / 0.000315` | `0.00057 / 0.0013 / 0.0013` | `0.000117 / 0.000227 / 0.000242` | `0.00021 / 0.000466 / 0.000496` | sparse MLP control |
| transition_routes_4 | `0.0182 / 0.1975 / 0.3526` | `0.0032 / 0.0819 / 0.1844` | `0.0026 / 0.1285 / 0.2714` | `0.0057 / 0.1573 / 0.2486` | dense `p64` hard-init LISTA |
| var_depth_gradient_4 | `0.0027 / 0.0021 / 0.0019` | `0.0110 / 0.0294 / 0.0318` | `0.0253 / 0.0332 / 0.0341` | `0.0287 / 0.0531 / 0.0562` | blockdiag hard-init LISTA |
| var_diamond_4 | `0.0062 / 0.0764 / 0.1281` | `0.0070 / 0.0987 / 0.1395` | `0.0321 / 0.1734 / 0.2195` | `0.1662 / 0.3434 / 0.3950` | blockdiag hard-init LISTA |
| var_l_shape_5 | `0.0064 / 0.0420 / 0.0467` | `0.0158 / 0.0205 / 0.0210` | `1.3976 / 5.1037 / 5.0609` | `1.4131 / 5.3622 / 5.3153` | dense `p64` hard-init LISTA |

- Interpretation of the fixed-`17` forecasting table:
  by lowest per-system `H1000`, the block-diagonal hard-init LISTA root is
  best on `7/17` systems, the dense `p64` hard-init LISTA root is best on
  `5/17`, and the sparse MLP control is best on `5/17`; the zero-`L1`
  ReLU control is not the `H1000` winner on any system in the locked packet.
  This is why the branch should not be written as a pure forecasting win for
  one root. The dense `p64` hard-init result is the causal basin-support win,
  the block-diagonal hard-init result is the stronger forecast-retaining
  companion, and the sparse MLP control still owns a nontrivial minority of
  systems on forecasting alone.

### April 14, 2026: fixed-17 LISTA phase-portrait supporting packet completed at `H1000`, `H3000`, and `H5000`

- Concrete result(s):
  generated the fixed-`17` LISTA visual packet under
  [fixed17_h1000_h3000_h5000_lista_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/fixed17_lista_phase_portraits_20260414/fixed17_h1000_h3000_h5000_lista_phase_portraits_manifest.json),
  writing `51` figures (`17` systems x `3` horizons), `17` per-system
  selection JSON files, and paired PNG/PDF outputs for every requested
  portrait. The generator scans collected transition-rich LISTA forecasting
  rows under `results/`, selects the lowest saved `H1000` best-periodic run
  per system, and reuses that run's saved `H1000` best-periodic mode for all
  requested horizons. The selected packet spans `8` LISTA roots:
  `lista_dense_softblock_signsplit_coherence_basin_partition` (`3` systems),
  `lista_blockdiag_signsplit_hardinit_basin_partition` (`6`),
  `lista_dense_basin_partition` (`2`),
  `lista_dense_softblock_signsplit_p64_hardinit_basin_partition` (`2`),
  `lista_blockdiag_sparsegroup_basin_partition` (`1`),
  `lista_blockdiag_basin_partition` (`1`),
  `lista_blockdiag_signsplit_basin_partition` (`1`), and
  `lista_blockdiag_double_basin_partition` (`1`).

- Result in experimental context:
  this closes the fixed-`17` visual-handoff gap with the same transition-rich
  shortlist as the live branch, while avoiding an artificial restriction to
  only the two promoted final-packet LISTA roots. Some systems still forecast
  best under earlier dense or block-diagonal provenance roots, while others
  are best under the later coherence, sparse-group, or hard-init follow-ups.

- Interpretation:
  the best available LISTA visuals are more heterogeneous than the locked
  causal comparison table. That heterogeneity is a presentation fact about the
  forecasting provenance on this branch, not a reason to reopen the causal
  shortlist or reinterpret the dense `p64` hard-init result as no longer the
  basin-support winner.

- Project implications:
  the paper appendix and senior-coauthor packet no longer have a missing
  fixed-`17` phase-portrait blocker. We can now show long-horizon qualitative
  behavior directly on the live shortlist, but because the chosen figures span
  multiple LISTA roots and sampling regimes, they must stay labeled as
  supporting visual context rather than new benchmark-comparison evidence.

- Next steps:
  choose a small subset of the `51` figures for the coauthor handoff, slides,
  and appendix, keep the locked causal claim anchored to the final comparison
  tables, and only use the new packet to illustrate qualitative trajectory
  behavior on representative systems.

### April 10, 2026: the working-budget zero-sparsity no-shrink control completed cleanly at default `dt`

- Concrete result(s):
  the zero-sparsity no-shrink control packet
  [transition_rich_zero_sparse_control_noshrink_20k_seed3_20260410](/home/mila/l/lia/skae/results/transition_rich_zero_sparse_control_noshrink_20k_seed3_20260410)
  finished as training array `9223056_[0-50]`, collect `9223057`, and resolve
  `9223058`. The pass-`0` collector wrote `51` rows across the fixed `17`
  systems with `17/17` good systems at default `dt`, and the resolver accepted
  every system at pass `0`. The system-median best-periodic forecasting values
  are `1.8317` at `H100`, `3.5797` at `H500`, and `4.1857` at `H1000`.

- Result in experimental context:
  this is a working-budget (`20k`, `3`-seed) zero-sparsity control screen on
  the fixed shortlist, using the no-shrink MLP encoder recipe to sharpen the
  branch's sparse-vs-zero-sparse-vs-LISTA framing without spending another
  locked-budget rerun first.

- Interpretation:
  the control is operationally fine at default `dt`, but it is not close to
  the locked packet's forecasting frontier. Removing encoder shrinkage and
  sparsity does not recover the final packet's long-horizon quality: the new
  no-shrink control stays in-band, yet it remains far weaker than the locked
  sparse MLP control and both promoted LISTA finalists.

- Project implications:
  this result strengthens the supporting claim that zero-sparsity controls are
  useful for causal contrast but are not new lead models for the basin-support
  branch. It also means the remaining live forecasting question is no longer
  default-`dt` viability for zero-sparsity MLPs; it is whether the selected
  deep-versus-near-separatrix forecasting split behaves the way the basin-
  identity narrative predicts.

- Next steps:
  let the corrected locked-budget tanh / no-shrink queue `9224111` emit its
  fresh task table and complete, then pair that control with the still-missing
  state-conditioned `H100/H500/H1000` evaluation so the final writeup can say
  something precise about where sparsity helps forecasting and where it does
  not.

### April 10, 2026: pass-`0` collect finished for the locked `10`-seed packet and rescue pass `1` is now underway

- Concrete result(s):
  pass-`0` forecasting collection for
  [transition_rich_basin_partition_final_seed10_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409)
  completed as `9211291`, with resolve `9211292` and auto-advance
  `9211747` also completed. The emitted rescue pass-`1` array is
  `9214917_*`. The collected system-median best-periodic `H1000` values are
  `0.0516` for `lista_blockdiag_signsplit_hardinit_basin_partition`,
  `0.0778` for `lista_dense_softblock_signsplit_p64_hardinit_basin_partition`,
  `0.0608` for `mlp_sparse_basin_partition_control`, and `0.0905` for
  `mlp_zero_sparse_basin_partition_control`, with all four roots currently at
  `17/17` good systems.

- Result in experimental context:
  this is the first multi-seed read from the locked `200k`, `10`-seed,
  fixed-`17` paper packet that compares the two promoted LISTA finalists
  against both MLP controls under one common forecasting protocol. It is the
  forecast half of the final confirmatory comparison, but it is still only the
  pass-`0` collection and therefore still missing rescued rows plus the
  state-level interpretability reduction.

- Interpretation:
  the new multi-seed forecasting read does not overturn the seed-`0`
  shortlist. Both promoted LISTA roots remain forecast-competitive at the
  larger budget; the block-diagonal hard-init root is currently the strongest
  pure forecasting model, the dense `p=64` hard-init root remains close, the
  matched sparse MLP control is competitive but not dominant, and the
  zero-`L1` ReLU ablation is currently weakest. This matters because it
  shows that the interpretability finalists were not selected by trading away
  forecasting entirely.

- Project implications:
  this pass strengthens confidence that the paper-facing support conclusions
  are compatible with the final training budget, but it does not yet change
  the main interpretability claim. There is still no newer state-level
  basin-support reduction for this packet, so the paper cannot yet replace the
  seed-`0` interpretability ranking with a locked multi-seed comparison.

- Next steps:
  let rescue pass `1` finish, rerun collection on the completed packet, then
  run the final interpretability reduction and paired LISTA-vs-control summary
  before changing the paper claim.

### April 9, 2026: the clean cross-root rerun completed and confirms the hard-init finalists

- Concrete result(s):
  the clean compatibility rerun
  [interpretability_rerun_compat_20260409](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409/interpretability_rerun_compat_20260409)
  completed as job `9211252` and wrote `4131` interpretability rows with `0`
  failures
  ([interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409/interpretability_rerun_compat_20260409/interpretability_summary.md),
  [manifest.json](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409/interpretability_rerun_compat_20260409/manifest.json)).
  On deep-basin states at `absolute:0.001`,
  `hardinit_packet_blockdiag_hardinit` beats its retrained base in
  `H(S|B)` (`1.4278 -> 1.3487`), `U_exact` (`0.7184 -> 0.7340`), `H(F|B)`
  (`0.1128 -> 0.1016`), own-basin projection damage (`25.5175 -> 7.7018`),
  and wrong-support damage (`0.7589 -> 0.3035`). The dense `p64` hard-init
  root remains the stronger forecasting / intervention tradeoff with
  `H(S|B)=0.7952`, `U_exact=0.8161`, `H(F|B)=0.0456`, `own/base=3.0430`, and
  `freeze/base=0.6715`. The matched sparse MLP control lands at
  `H(S|B)=1.0922`, `U_exact=0.7757`, `H(F|B)=0.0127`, `own/base=15.5769`, and
  `freeze/base=1.0690`.

- Result in experimental context:
  this rerun is the fixed version of the first common paper-facing reduction
  across the `v5` shortlist leaders, the `v6` identifiability follow-ups, the
  matched sparse MLP control, and the hard-init packet's retrained base /
  variant pairs. It uses the state-level support protocol from
  [interpretability_study_plan.md](/home/mila/l/lia/skae/docs/planning/interpretability_study_plan.md)
  and the fixed root bundle chosen from
  [basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md)
  and
  [transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md).
  The original run was invalid because of launch / checkpoint-compatibility
  bugs; this rerun is the first scientifically usable version of that
  comparison.

- Interpretation:
  the rerun does not change the scientific ranking; it validates it. The
  forecast-retaining interpretability finalist remains the block-diagonal
  hard-init root, while the dense `p64` hard-init root remains the stronger
  forecasting / intervention tradeoff. The sparse MLP control is not the
  branch winner on the main objective because its exact-support compression
  and intervention robustness are still weaker on the selected slice. In line
  with the April 9 notes on possibly shared basin eigenstructure, the raw
  operator-separation ratios are supporting context rather than the decisive
  criterion; support separation and support-conditioned interventions remain
  the primary read.

- Project implications:
  the seed-`0` ranking blocker is gone. There is no evidence here that
  justifies another broad LISTA sweep before the fair control comparison
  finishes. The remaining paper decision is now robustness and effect size,
  not method discovery.

- Next steps:
  let the locked `200k`, `10`-seed confirmatory packet finish, rescue the
  node-local ECC failures through the existing pass-`0` resolve / watcher
  chain, then run the final interpretability reduction and the paired
  LISTA-vs-sparse-MLP-vs-zero-sparsity-MLP summary on the same selected
  support slice.

### April 9, 2026: `v5` and `v6` are fully reduced; `v5 p64` is the current shortlist winner and `v6` clarifies the forecast-versus-interpretability tradeoff

- Concrete result(s):
  all submitted fixed-`17` shortlist packets `v1-v6` are now complete through
  arrays, collect, resolve, and interpretability reduction. The key new
  artifacts are the `v5` forecasting summary
  [forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v5/collect_pass0/forecasting_summary.md),
  the `v5` interpretability summary
  [interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v5/interpretability_pass0/interpretability_summary.md),
  the `v6` forecasting summary
  [forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v6/collect_pass0/forecasting_summary.md),
  and the `v6` interpretability summary
  [interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v6/interpretability_reduce_pass0/interpretability_summary.md).
  The best new exact-support root is
  `lista_dense_softblock_signsplit_p64_basin_partition` from `v5` with deep-
  basin `absolute:0.001` metrics `mean H(S|B)=0.7719`,
  `mean U_exact=0.8064`, `mean H(F|B)=0.0521`, `mean own/base=14.6200`, and
  `mean wrong/base=3007.3059`, while still keeping `16/17` systems below the
  `H1000` good-forecast gate. The best forecast-retention root across these
  two packets is `lista_blockdiag_signsplit_basin_partition` from `v5`
  (`H1000 system-median = 0.0119`, `17/17` good systems), and the strongest
  `v6` forecasting root is
  `lista_dense_softblock_signsplit_coherence_basin_partition`
  (`H1000 system-median = 0.0585`, `17/17` good systems).

- Result in experimental context:
  these are exactly the next two plan-driven tiers after the earlier
  structured / soft-block sweep. `v5` tested sign-split codes, deeper LISTA
  refinement, block-count expansion, and latent-size sweeps; `v6` tested the
  two remaining identifiability levers called out in the design notes:
  restraining the pre-code MLP and reducing decoder ambiguity with coherence.
  Both packets ran on the fixed `17`-system shortlist at the working `20k`
  budget with saved forecasting and interpretability reductions. In the `v6`
  loss, coherence means penalizing off-diagonal Gram-matrix energy among
  normalized decoder atoms, after collapsing sign-split atom pairs back to one
  effective atom.

- Interpretation:
  `v5` is the first shortlist tier that materially compresses exact supports
  within basin while keeping the model operational on almost all systems. The
  `p64` dense sign-split root is the current best interpretability candidate.
  `v6` then sharpens the mechanism read: coherence helps forecasting, while
  linear encoders help exact-support compression, but the linear-encoder gain
  is not large enough to justify its forecasting penalty, and none of the
  `v6` roots beats `v5 p64` on the combined frontier. At the same time, the
  canonical exact-support intervention still hurts even in the correct basin,
  so the branch still lacks evidence for one operative canonical chart per
  basin.

- Project implications:
  the paper now has a substantially stronger exact-support-compression
  positive than it had before `v5`, but the honest claim is still weaker than
  “one exact support equals one local chart.” The evidence now supports a
  shortlist story closer to “LISTA can make exact supports substantially more
  basin-specific and reusable” while leaving open whether the final paper
  should land at exact-support reuse or support-family / dominant-group
  alignment. The matched sparse MLP control already exists in `v1`, so the
  next causal comparison can now be written from completed runs rather than
  from another queue.

- Next steps:
  compare the `v5 p64` root, the best forecast-retaining `v5` block-diagonal
  root, and the matched `v1` sparse MLP control on the same study-plan
  metrics. Only run another LISTA packet if that comparison still leaves a
  clear exact-support claim within reach; if so, restrict it to the remaining
  high-value plan items around `v5 p64`: adaptive thresholds, group-aware
  shrinkage, richer reset triggers, and dictionary-tied or hybrid pre-codes.
  Otherwise, lock the paper narrative around the strongest completed LISTA
  root, finish the remaining evaluation-only diagnostics, and state the
  canonical-chart limitation explicitly.

### April 9, 2026: restrained-precode and decoder-coherence LISTA ablations were implemented and queued as `v6`

- Concrete result(s):
  the training CLI now accepts `--lista_final_op sign_split`, exposes a
  boolean `--lista_linear_encoder` override, and exposes a
  `--decoder_coherence_weight` override. `LISTAKM` now computes a normalized
  decoder-coherence penalty that collapses sign-split atom pairs before
  measuring off-diagonal Gram energy. Focused validation completed with `109`
  targeted tests passing on a compute node. The new seed-`0` packet
  [transition_rich_basin_partition_20260409_seed0_smoke_v6](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v6)
  was launched with default array `9203315_[0-101]`, pass-`0` collect
  `9203316`, pass-`0` resolve `9203317`, and pass-`0` interpretability reduce
  `9203318`.

- Result in experimental context:
  this is the next direct follow-through on
  [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md)
  after the sign-split / latent-sweep `v5` tier. The six new roots isolate
  two remaining plan-item axes on top of the strongest current sign-split
  recipes:
  `lista_blockdiag_signsplit_linear_encoder_basin_partition`,
  `lista_blockdiag_signsplit_coherence_basin_partition`,
  `lista_blockdiag_signsplit_linear_encoder_coherence_basin_partition`,
  `lista_dense_softblock_signsplit_linear_encoder_basin_partition`,
  `lista_dense_softblock_signsplit_coherence_basin_partition`, and
  `lista_dense_softblock_signsplit_linear_encoder_coherence_basin_partition`.
  This packet is deliberately causal: it tests whether the pre-code MLP is
  bypassing sparse-code identifiability and whether dictionary ambiguity is
  contributing directly to within-basin support fragmentation.

- Interpretation:
  at this stage there was no scientific `v6` result yet; this entry records
  the implementation and launch event. The important read at that moment was
  that the branch no longer had a tooling gap at the next two design-note
  items, and the queued sign-split packet was no longer at risk of failing
  immediately on an unrecognized CLI value.

- Project implications:
  the shortlist is now queued through the next meaningful interpretability
  levers without expanding system scope or training budget. If `v6` reduces
  exact-support entropy while preserving basin purity, the paper can still
  chase an exact-support chart story. If it does not, the branch will have
  much stronger evidence that the honest NeurIPS-facing claim is at the
  support-family / dominant-group level.

- Next steps:
  this queue-era next step is now complete; see the newer April 9 completed-
  results entry above for the reduced `v5` / `v6` read and the current
  shortlist decision.

### April 9, 2026: the study-plan metric stack now includes support families and support projections, and the sign-split / latent-sweep `v5` packet was launched

- Concrete result(s):
  the interpretability reducer now adds greedy Jaccard support-family metrics
  plus deep-basin canonical-support projection diagnostics, and focused
  validation completed with `80` targeted tests passing on a compute node. The
  upgraded reducer smoke is saved under
  [transition_rich_interpretability_smoke_20260409/native_seed0_v2_family_projection](/home/mila/l/lia/skae/results/transition_rich_interpretability_smoke_20260409/native_seed0_v2_family_projection),
  and the sign-split / deeper-LISTA / latent-size / block-count packet
  [transition_rich_basin_partition_20260409_seed0_smoke_v5](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v5)
  is now live with default array `9203114`, pass-`0` collect `9203116`,
  pass-`0` resolve `9203117`, and pass-`0` interpretability reduce `9203118`.
- Result in experimental context:
  this carries the branch forward on both sides of the loop the user asked
  for. On the diagnosis side, it implements the next study-plan metrics that
  were still missing after exact-support entropy and purity: support families
  and counterfactual support interventions. On the training side, it finally
  gets the planned item-`7/8` packet onto the cluster after the earlier
  admission failures.
- Interpretation:
  the native-trio smoke is already informative. Deep-basin exact supports are
  still fragmented (`H(S|B)` stays around `2.19-2.30` at `absolute:0.001`),
  but the family view is much cleaner (`H(F|B) ~= 0.2388` block-diagonal,
  `0.1322` dense). At the same time, projecting to the canonical exact support
  of the correct basin still hurts one-step prediction (`own/base > 1`), even
  though wrong-basin projections are catastrophically worse. That is evidence
  for basin-selective support structure, but not yet for one exact canonical
  support acting as the local chart.
- Project implications:
  the paper story should currently lean toward support families / dominant
  groups rather than one exact support per basin unless `v5` changes that. The
  next packet is now well aligned with that gap: sign-split codes, more LISTA
  refinement, and latent/block-count sweeps are exactly the planned tools most
  likely to collapse exact-support fragmentation.
- Next steps:
  let `v5` finish its default pass, reduce it with the upgraded study-plan
  metrics, and check whether the new packet lowers `H(S|B)` toward the already
  stronger family-level read. If it does, compare the winning LISTA arms
  against the matched MLP control on the same metrics.

### April 9, 2026: structured hard-block and soft block-sparse LISTA ablations are now implemented and queued as the next shortlist packet

- Concrete result(s):
  the transition-rich packet now includes
  `lista_dense_softblock_basin_partition`,
  `lista_dense_softblock_strong_basin_partition`,
  `structured_lista_temporal_basin_partition`,
  `structured_lista_entropy_temporal_basin_partition`, and
  `structured_lista_dominance_temporal_basin_partition`. The model/config
  stack now supports an explicit dense-`K` off-block penalty, and the
  study-plan reducer now handles structured/global-plus-basin and soft-block
  group layouts. Validation completed with `98` targeted tests passing on a
  compute node, plus CLI smokes under
  [/tmp/skae_softblock_smoke/20260409-014101](/tmp/skae_softblock_smoke/20260409-014101)
  and
  [/tmp/skae_structured_smoke/20260409-014101](/tmp/skae_structured_smoke/20260409-014101).
  The new packet
  [transition_rich_basin_partition_20260409_seed0_smoke_v4](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v4)
  is queued through the default pass and first rescue pass with jobs
  `9202904_[0-84]`, `9202905`, `9202906`, `9202907_[0-84]`, `9202908`,
  `9202909`, plus reducer jobs `9202910` and `9202911`.
- Result in experimental context:
  this is the first execution of the next two unrun items in
  [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md)
  after the dense reset-policy tier: a soft block-sparse single-`K` family
  and a hard structured basin-block family with within-trajectory group
  persistence.
- Interpretation:
  the branch is no longer blocked on missing implementation support for those
  matrix families. Any negative read from `v4` will now be evidence against
  these ablations on the fixed shortlist, not evidence that they were never
  run. The queue caveat is operational rather than scientific: the launcher
  hit the cluster submit cap before it could emit later rescue passes.
- Project implications:
  the ablation loop has now moved past dense-versus-block-diagonal and
  reset-policy tuning into the two main structural alternatives still missing
  from the shortlist evidence. The next reduction can finally test whether
  exact-support fragmentation is better attacked by softening `K` into near-
  block structure or by explicitly imposing basin-wise latent groups and
  temporal persistence.
- Next steps:
  let `v4` finish its default pass, reduce it with the chained study-plan
  summaries, and defer any deeper rescue ladder until enough older queued jobs
  clear to restore submit-budget headroom.

### April 9, 2026: dynamics-aware reencoding and projection-gap reset ablations are now implemented locally and submitted as the next fixed-`17` packet

- Concrete result(s):
  the core evaluation stack now supports dynamics-aware latent warm starts at
  reencode time, projection-gap-triggered resets, saved rollout reset
  diagnostics, and a `best_reset` summary. Local validation completed with
  `75` focused tests passing on a compute node, plus a full CLI smoke run on
  `gated_local_linear` that produced the new `event_proj_0p05` mode and saved
  artifacts under
  [/tmp/skae_eval_smoke/20260409-011940](/tmp/skae_eval_smoke/20260409-011940).
  The follow-on fixed-`17` packet
  [transition_rich_basin_partition_20260409_seed0_smoke_v3](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v3)
  is now live with launcher `9202783`, default array `9202785_[0-50]`,
  collector `9202786`, final collector `9202804`, final resolver `9202805`,
  and reducer jobs `9202814` / `9202815`.
- Result in experimental context:
  this is the first time the highest-priority remaining design-note items from
  [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md)
  are actually exposed as packet-level axes rather than as prose TODOs. The
  new `v3` wave isolates the dense-arm order from the note itself:
  projection-gap trigger first, then dynamics-aware LISTA reencoding, then
  dynamics-aware HyperLISTA reencoding.
- Interpretation:
  the branch is no longer blocked on missing reset-policy infrastructure.
  From this point on, a negative read on reset-trigger or dynamics-aware
  ablations will be scientific evidence, not an implementation gap.
- Project implications:
  the live interpretability loop can now test whether within-basin support
  fragmentation is really a reencoding-policy problem before spending more
  budget on larger architectural moves. It also means the next shortlist
  reduction should compare `best_reset`, not only `best_periodic`, whenever an
  arm actually includes event-triggered evaluation.
- Next steps:
  let `v3` complete its first default-`dt` pass, inspect whether the
  dynamics-aware roots change `best_reset` and support-fragmentation behavior
  relative to `v1` / `v2`, and reduce the new packet with the chained
  study-plan summaries as soon as pass `0` and the final collect finish.

### April 9, 2026: study-plan state-level metrics confirm basin purity but not exact-support uniqueness on the historical native trio

- Concrete result(s):
  the new reducer under
  [interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_interpretability_smoke_20260409/native_seed0/interpretability_summary.md)
  successfully reduced `6` historical native-trio LISTA checkpoints (`3`
  systems x `2` roots x seed `0`) from
  [collect_pass6/forecasting_rows.csv](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/collect_pass6/forecasting_rows.csv)
  with zero failures and `162` metric rows. On that subset:
  - deep-basin `H(B|S)` is `0.0000` for both dense and block-diagonal LISTA
    across the reported top-`k` and several thresholded support definitions
  - deep-basin `H(S|B)` stays high on thresholded supports, for example
    `1.7070` for block-diagonal LISTA at `relative:0.1` and `1.8196` for dense
    LISTA at the same setting
  - aggressive top-`k` masking raises exact-support concentration, with dense
    `topk:4` reaching deep-basin `U_exact = 0.8299`, `NMI = 0.8412`, and
    block-diagonal `topk:8` reaching `U_exact = 0.7259`, `NMI = 0.7987`
  - thresholded support-conditioned operator separation is still only modestly
    above `1`, for example all-state `operator_between_over_within = 1.6693`
    for block-diagonal LISTA and `1.6422` for dense LISTA at `relative:0.1`
- Result in experimental context:
  this is the first run of the new study-plan reducer, and it was executed on
  the historical native-trio LISTA checkpoints only to validate the metric
  stack before spending new `20k` queue budget on fresh ablations and the MLP
  control.
- Interpretation:
  the new state-level metrics agree with the older recurring-support reduction
  on the qualitative point that current LISTA supports are basin-pure much
  more readily than they are mechanism-clean. Deep inside a basin, support can
  identify basin almost perfectly while still fragmenting into many exact
  supports. Dense LISTA currently looks somewhat sharper on deep top-`k`
  exact-support concentration, while block-diagonal LISTA remains slightly
  better on thresholded operator-separation ratios.
- Project implications:
  the next live ablation wave should optimize for lower `H(S|B)`, higher
  `U_exact`, and better within-basin support persistence, not just low
  `H(B|S)`. The matched MLP control is now even more important, because the
  remaining question is not whether LISTA supports are pure at all, but
  whether LISTA can make them materially more canonical and operator-aligned
  than a non-LISTA encoder on the same fixed shortlist.
- Next steps:
  finish the live `20k` seed-`0` five-root packet, reduce it with the same
  state-level metrics, and only then decide whether to spend the next queue
  budget on more seeds or on additional LISTA support-compression ablations.

### April 8, 2026: fixed-`17` LISTA basin-support reduction is purity-positive but local-linearity-negative

- Concrete result(s):
  the completed local reduction under
  [basin_support_metrics_20260408_v3](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/basin_support_metrics_20260408_v3)
  evaluated the latest fixed-shortlist LISTA checkpoints from
  [collect_pass6/forecasting_rows.csv](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/collect_pass6/forecasting_rows.csv)
  on the paper-facing recurring-support protocol. Across the `17 x 2 = 34`
  LISTA runs:
  - mean support-group purity is `0.9883` for
    `lista_blockdiag_basin_partition` and `0.9961` for
    `lista_dense_basin_partition`
  - mean retained-trajectory coverage is `0.8729` for block-diagonal LISTA and
    `0.8787` for dense LISTA
  - `15/17` systems clear the `0.60` coverage gate for both roots
  - local `H=20` NRMSE beats global `H=20` NRMSE on `0/17` systems for both
    roots
  - local `H=20` NRMSE beats the shuffled baseline on `0/17` block-diagonal
    systems and `1/17` dense systems; that lone dense win is
    `claude:duffing_triple_well`, where coverage is only `0.1172`
  - the main low-coverage exceptions are `claude:duffing_triple_well`
    (`0.1094` / `0.1172`) and `multiwell_strong_transition`
    (`0.3984` / `0.4531`)
- Result in experimental context:
  this is the first full fixed-shortlist reduction that matches the March
  recurring-support local-linearity protocol but applies it to the April 7/8
  LISTA shortlist packet rather than to the earlier three-system screening
  family. The reducer uses the latest collected row per
  `(root_label, system_key, seed)`, regenerates the shared held-out trajectory
  corpus from checkpoint-compatible environments, measures support-group purity
  and retained coverage, and compares local versus global versus shuffled
  `H=20` ridge fits.
- Interpretation:
  the LISTA shortlist now has strong evidence for basin-pure recurring-support
  structure on most systems, and the retained support groups are usually broad
  enough to survive the `0.60` coverage gate. But those same recurring
  supports do **not** currently justify the stronger statement that
  support-conditioned local linear maps explain held-out `20`-step dynamics
  better than a single global map.
- Project implications:
  the live branch can now be written as a basin-purity / recurring-support
  reuse result for LISTA, not as a local-linearity-mechanism win. That makes
  the matched standard-MLP control even more important: the next causal
  question is whether the same high-purity, high-coverage support structure is
  actually stronger for LISTA than for a non-LISTA encoder on the same packet.
- Next steps:
  run the matched standard-MLP control through the same reducer, add the
  resulting contrast to the fixed-`17` branch summary, and treat the new
  local-linearity numbers as a negative supporting diagnostic unless a later
  ablation changes that read.

### April 8, 2026: overnight rescue continuation broadened the default-`dt` LISTA read, but exposed an `env_dt` collection bug

- Concrete result(s):
  the six dependency-chained overnight rescue arrays
  `9193426`, `9193429`, `9193432`, `9193435`, `9193438`, and `9193441`
  completed between `2026-04-08 02:16 EDT` and `2026-04-08 20:24 EDT`,
  together with collectors / resolvers `9193427-9193443`, under
  [transition_rich_basin_partition_20260407](/network/scratch/l/lia/skae/transition_rich_basin_partition_20260407).
  These jobs were launched to halve `dt` whenever an arm failed
  `H1000 best-periodic < 50`, but the collected scientific read is different:
  - [collect_pass0](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/collect_pass0/forecasting_summary.md)
    already reports `17/17` good systems at `H1000` for both
    `lista_dense_basin_partition` and `lista_blockdiag_basin_partition` on the
    fixed shortlist
  - all `192/192` overnight rescue-pass training runs still satisfy
    `H1000 best-periodic < 50` at row level, even though they were supposed to
    be halved-`dt` rescues
  - across the rescue-only reruns, all `16/16` rerun systems remain below the
    same gate by system median for both LISTA roots
  - the final combined collector
    [collect_pass6](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/collect_pass6/forecasting_summary.md)
    still reports `17/17` good systems at `H1000`, with system-median
    `H1000 best-periodic` `0.0653` for dense LISTA and `0.0832` for
    block-diagonal LISTA
  - representative same-`dt` improvements across the repeated `20k` reruns are:
    dense `claude:checkerboard_potential` `0.3053 -> 0.0198`,
    dense `claude:cal_square_4` `0.0641 -> 0.00154`,
    dense `claude:var_l_shape_5` `0.0409 -> 0.00579`,
    block-diagonal `claude:var_l_shape_5` `0.9626 -> 0.0967`,
    block-diagonal `claude:cal_high_cross_3` `0.8760 -> 0.1727`,
    block-diagonal `claude:transition_routes_4` `0.5244 -> 0.1574`, and
    block-diagonal `gated_transfer_linear` `0.7945 -> 0.4899`
- Result in experimental context:
  the overnight chain was intended to test whether any fixed-shortlist LISTA
  arm needed smaller `dt` after the default-source collector ran. Instead, it
  exposed a collection / resolver plumbing issue:
  `32/34` rows in
  [collect_pass0/forecasting_rows.csv](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/collect_pass0/forecasting_rows.csv)
  and `224/226` rows in
  [collect_pass6/forecasting_rows.csv](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/collect_pass6/forecasting_rows.csv)
  were missing `env_dt`, even though the run configs stored the correct values.
  The cause was that
  [collect_forecasting_roots.py](/home/mila/l/lia/skae/tools/collect_forecasting_roots.py)
  only read built-in env `DT` fields and ignored
  `GATED_LOCAL_LINEAR`, `GATED_TRANSFER_LINEAR`, and `CLAUDE_CATALOG`.
  Because the resolver could not see those `dt` values, it kept reissuing the
  same default-`dt` task tables for the remaining arms rather than true
  halved-`dt` rescues.
- Interpretation:
  the scientifically relevant result is stronger default-`dt` evidence, not a
  smaller-`dt` rescue effect. On the current one-seed LISTA packet, smaller
  `dt` is not the practical bottleneck: the user-facing `H1000` gate is
  already satisfied on the full fixed shortlist for both dense and
  block-diagonal LISTA. The overnight `20k` reruns also suggest that the
  forward diagnostic recipe is already robust enough to stay inside that coarse
  forecasting gate on the same shortlist.
- Project implications:
  for the basin-separability branch, step-size rescue is now further
  de-emphasized. The live interpretability story should move even more clearly
  toward basin-support reduction and the matched standard-MLP comparison,
  because the overnight jobs neither open nor close the central causal claim;
  they only reinforce that default `dt` is already adequate on the one-seed
  LISTA shortlist. Operationally, do not spend more queue budget on pass `7`
  or further `dt` halving from this chain. The collector bug is fixed locally,
  so future collection will preserve `env_dt` for `gated_*` and `claude:*`
  arms.
- Next steps:
  treat the April 8 rescue chain as scientifically closed, recollect /
  re-resolve only if we want a clean selected-`dt` table after the collector
  patch, and spend the next queue budget on basin-separability evaluation and
  the matched standard-MLP control rather than on more `dt` rescue.

### April 7, 2026: initial default-`dt` LISTA basin-partition read on the native trio

- Concrete result(s):
  the first six manually evaluated native-system checkpoints from the live
  `17 x 2 x 1` LISTA sweep all already satisfy the `dt`-rescue gate at each
  system's default step size under `H1000 best-periodic < 50`.
  - `gated_local_linear`
    - dense: `H1000 best-periodic = 1.5437e-03`
    - block-diagonal: `H1000 best-periodic = 1.7160e-03`
  - `gated_transfer_linear`
    - dense: `H1000 best-periodic = 1.2213`
    - block-diagonal: `H1000 best-periodic = 6.7939e-01`
  - `multiwell_strong_transition`
    - dense: `H1000 best-periodic = 5.8041e-01`
    - block-diagonal: `H1000 best-periodic = 4.4335e-02`
  - the manual diagnostic root is
    [manual_eval](/network/scratch/l/lia/skae/transition_rich_basin_partition_20260407/manual_eval)
- Result in experimental context:
  these are checkpoint-by-checkpoint manual evaluations run on compute nodes
  while the live arrays are still training and before any natural
  `evaluation_best/.../rollout_artifacts.pt` outputs have appeared from the
  live sweep. The purpose was to test the new default-`dt` then halve-`dt`
  rescue rule early on the native trio, and to use the new rollout-diagnostics
  stack to identify failure modes beyond MSE.
  - `gated_local_linear`:
    both variants pass the gate, but `no_reencode` remains poor and predicted
    basin crossing fraction is `1.0` versus true `0.0`; block-diagonal raises
    basin endpoint accuracy from `0.03` to `0.34` but inflates basin
    transition-count MAE from `4.76` to `102.56`
  - `gated_transfer_linear`:
    both variants pass the gate; block-diagonal improves best-periodic `H1000`,
    basin transition-count MAE (`14.79` versus `30.12`), and support-switch
    count (`55.33` versus `303.64`), but both still predict basin crossing
    fraction `1.0` versus true `0.15`
  - `multiwell_strong_transition`:
    both variants pass the gate; block-diagonal sharply improves best-periodic
    `H1000` (`4.4335e-02` versus `5.8041e-01`), while dense is slightly better
    on basin transition-count MAE (`10.31` versus `12.89`) and support-switch
    count (`30.06` versus `33.93`)
- Interpretation:
  for the branch's main claim, the key positive result is that support groups
  remain basin-pure in every inspected native LISTA arm. Default `dt` is not
  the current blocker, and support contamination is not the current failure
  mode. The main remaining weakness is secondary: free rollouts still switch
  supports too often and invent too many basin crossings relative to the true
  native dynamics.
- Project implications:
  the paper should write this native-trio read around basin separation first:
  LISTA is already producing basin-pure support groups on the branch's most
  important native systems. Transition-fidelity diagnostics still matter, but
  they should be presented as a secondary limitation layered on top of an
  encouraging support-partition signal rather than as the branch headline.
- Next steps:
  wait for the first natural live-sweep `rollout_artifacts.pt` outputs, then
  extend the same support-view reduction across the Claude subset and the
  matched standard-MLP control. The next branch-level summary should count
  systems where LISTA gives cleaner basin separation than the non-LISTA
  baseline, not just systems with acceptable best-periodic error.

### April 7, 2026: implemented Claude-catalog audit

- Concrete result:
  [docs/planning/claude_catalog_audit_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_audit_20260407.md)
  now verifies that the current worktree has `112` registered Claude-catalog
  systems, `83` systems covered by the combined grounded fast screen, `29`
  implemented but unscreened systems, `12` accepted systems after a completed
  `15`-system priority screen, and an `8`-system strict-crossing subset within
  that accepted pool.
- Context:
  this audit was needed because the older
  [claude_transition_rich_catalog.md](/home/mila/l/lia/skae/docs/planning/claude_transition_rich_catalog.md)
  note claimed `44 confirmed passing` systems, which was strong enough to
  affect benchmark-positioning decisions if true.
- Interpretation:
  the implemented Claude catalog is real and useful, and it is no longer just a
  speculative backlog: several tuned controls and two hybrid mechanisms now
  survive the exact same fast screen. The retune pass also rescued
  `var_depth_gradient_4` into the strict-crossing core and `cal_hexagon_6`
  into the relaxed-accept subset. The stricter claim now needs one more split:
  `8` of the accepted systems keep every basin inside the strict crossing band,
  while `4` more survive through the relaxed crossing gate. Repeated official
  re-screens still leave `cal_octagon_8` outside the pool, so the high-basin
  control frontier is real rather than papered over.
  The
  saved artifacts still do not support treating the catalog as a validated
  large benchmark packet.
- Project implications:
  the paper should not promise a repaired `44`-system implemented benchmark.
  It can legitimately cite the grounded `12`-system Claude-catalog accepted
  pool, with an `8`-system strict core, as provenance for why the chosen
  Claude subset is technically grounded. But the audit no longer defines the
  active experiment scope by itself; forward branch experiments are restricted
  to the fixed `14` chosen Claude systems.
- Next steps:
  use the audit to annotate the chosen Claude subset by status, then plan or
  interpret runs only on the fixed `17`-system branch shortlist rather than
  reopening broader catalog or conceptual-inventory selection.

## Hard-Init Seed-0 Reduction (2026-04-09)

- Concrete result(s):
  the hard-init seed-`0` packet completed cleanly under
  [transition_rich_basin_partition_hardinit_seed0_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409),
  with forecasting summary in
  [forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409/collect/forecasting_summary.md)
  and interpretability summary in
  [interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409/reduce/interpretability_summary.md).
  `failures.json` is empty. On the study-plan default read
  (`absolute:0.001`, deep subset), the block-diagonal hard-init variant
  improves `H(S|B)` (`1.4297 -> 1.3493`), `U_exact` (`0.7181 -> 0.7447`),
  `H(F|B)` (`0.1129 -> 0.1018`), own-basin canonical projection ratio
  (`25.5197 -> 7.7018`), wrong-support ratio (`0.7599 -> 0.3034`), and raw
  operator-family separation (`1.8908 -> 2.4271`). On the same read, the
  dense `p64` hard-init variant keeps `H(B|S)=0`, nudges `U_exact`
  (`0.8070 -> 0.8161`) and `H(F|B)` (`0.0485 -> 0.0456`) in the right
  direction, but slightly worsens deep `H(S|B)` (`0.7624 -> 0.7959`) and
  all-state support compression. Forecasting-wise, dense `p64` hard-init is
  the clearer positive (`H1000` system-median best `0.1358 -> 0.0794`);
  block-diagonal hard-init is nearly neutral (`0.0800 -> 0.0841`) while
  improving the good-system count (`16/17 -> 17/17`).
- Result in experimental context:
  this packet was meant to test whether oversampling hard initial conditions
  near separatrices helps the current `v5` sign-split anchors learn cleaner
  basin support structure. The answer is not uniform across roots: the
  block-diagonal arm benefits more on the raw interpretability metrics,
  whereas the dense `p64` arm benefits more on forecasting and
  canonical-support counterfactual behavior.
- Interpretation:
  hard initialization looks like a real mechanism for improving the
  block-diagonal root's within-basin support consolidation and local-support
  intervention stability. For the dense `p64` root, the main positive is not
  a clean raw support-compression win; it is that the model forecasts better
  and its deep-basin canonical-support interventions are much less damaging
  even though the raw all-state support metrics do not improve consistently.
- Project implications:
  the branch should not treat hard-init as a blanket all-roots success and
  automatically promote the prepared `204`-run `3`-seed packet unchanged.
  The result is promising enough to keep hard-init in the paper-facing
  comparison, but the next decision should be variant-specific and grounded in
  the queued cross-root rank order rather than in this packet alone.
- Next steps:
  let the queued cross-root phase-`1` reducer finish, compare both hard-init
  variants against the strongest `v5`, `v6`, and matched `v1` MLP roots on the
  same metrics, then decide whether any hard-init variant deserves a `3`-seed
  robustness run. Keep the phase-`2` basis-aware rerun separate until the
  alignment-aware reducer extensions are implemented.

## Queue Status

- As of `2026-04-28 16:40 EDT`, the matched-dimension LISTA-SB fairness
  sensitivity is staged as queue job `9395415` using
  [scripts/queue_transition_rich_lista_sb_p256_hardinit_fairness.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_lista_sb_p256_hardinit_fairness.sh).
  The launcher will create
  `results/transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428/task_tables/transition_rich_lista_sb_p256_hardinit_fairness.tsv`
  on a compute node, then submit a `255`-task `%64` GPU array for
  `lista_dense_softblock_signsplit_p256_hardinit_basin_partition`, followed by
  forecasting collection and a `topk:8` self-routed forecasting queue. This is
  a reviewer-facing sensitivity for the current `d_z=64` LISTA-SB row, not a
  replacement for the existing paper row.

- As of `2026-04-28 16:27 EDT`, Table 2 five-model completion work is queued.
  Existing `200k` hard-init MLP-control seeds `0`--`9` are being evaluated for
  non-oracle self-routed forecasting by shard jobs `9395314`--`9395319`; merge
  job `9395320` will write
  [results/transition_rich_self_routed_forecasting_hardinit_mlp_controls_seed0to9_20260428/self_routed_forecasting_rows.csv](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_hardinit_mlp_controls_seed0to9_20260428/self_routed_forecasting_rows.csv).
  Dependent job `9395334` will refresh
  [per_system_paired_tests.json](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/per_system_paired_tests.json)
  after that merge.
  The five-model training backfill is staged as queue job `9395321`; it will
  generate and submit a `433`-task `%64` GPU array once the expanded user job
  count is low enough. That array covers seeds `10`--`14` for `LISTA-SB`,
  `LISTA-BD`, Dense MLP no-shrink, Sparse MLP, and Sparse MLP-BD, plus the
  `8` missing seed-`0`--`9` hard-init MLP-control rows needed for complete
  fixed-`17` coverage.

- As of `2026-04-28 15:03 EDT`, the corrected Dysts `seq_len=10`,
  `n=15`, H<=60K pipeline is training but not yet evaluated. Chunk 1 is job
  `9392814` and chunk 2 is job `9393590`; both arrays still have running and
  pending child tasks under their `%64` throttles. The original Dysts
  orchestrator `9392878` was canceled before it submitted chunk 2 or the
  invalid H100K/full-cache eval path; replacement orchestrator `9393138`
  submitted chunk 2 as `9393590` and will submit
  `scripts/queue_dysts_long_horizon_eval.sh` with
  `OUTPUT_TAG=dysts_long_horizon_h5k_to_h60k_seq10`,
  `DYSTS_CACHE_PROFILE=long60`, and horizons
  `5000 10000 20000 30000 40000 50000 60000` after both training arrays drain.
  Expected eval results path:
  [results/dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428).

- Table 2 seed expansion status as of `2026-04-28`: job array `9392598`
  completed `255/255` seed-`10`--`14` transition-rich basin-partition tasks,
  but this was the wrong scope for the paper-facing expansion. Its task table
  under
  [results/transition_rich_basin_partition_seed10to14_20260428/task_tables/transition_rich_basin_partition.tsv](/home/mila/l/lia/skae/results/transition_rich_basin_partition_seed10to14_20260428/task_tables/transition_rich_basin_partition.tsv)
  covers only the three original Table 2 roots and `num_steps=20000`. The
  requested five-model, `200k` seed-`10`--`14` expansion with Sparse MLP and
  Sparse MLP-BD is still outstanding.

- April 26 documentation pass: no new SLURM jobs were submitted and no live
  queue state was rechecked. The active documentation state is now the
  evidence-first map in
  [PAPER_EXPERIMENT_EVIDENCE_MAP.md](/home/mila/l/lia/skae/docs/PAPER_EXPERIMENT_EVIDENCE_MAP.md);
  the last recorded paper-critical queue status remains
  `2026-04-25 19:25 EDT`.

- As of `2026-04-25 19:25 EDT`, direct periodic-support-refresh smoke
  validation and the full fixed-`17` seed-`0` LISTA-only science shards are
  complete. Smoke job `9361455` wrote `32` ok rows and `0` failures under
  [results/periodic_support_refresh_smoke_20260425_cal_square](/home/mila/l/lia/skae/results/periodic_support_refresh_smoke_20260425_cal_square).
  Dense LISTA shard `9361464` completed with exit `0:0`, `16/16` specs,
  `34,440` rows, and `0` failures. Blockdiag LISTA shard `9361465` completed
  with exit `0:0`, `17/17` specs, `38,280` rows, and `0` failures. Merge job
  `9361470` is pending on scheduler priority for consolidated artifacts. The
  packet uses
  [forecasting_rows.csv](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/collect_pass0/forecasting_rows.csv),
  seed `0`, support definitions `absolute:0.001,topk:8,relative:0.1`,
  re-encode periods `1,10`, start modes `target_entry,post_start`, and writes
  to
  [results/periodic_support_refresh_fixed17_seed0_20260425](/home/mila/l/lia/skae/results/periodic_support_refresh_fixed17_seed0_20260425).
  This is the active outstanding mechanism test for the claim that periodic
  re-encoding refreshes support after basin entry and routes later Koopman
  evolution through different active coordinates.

- As of `2026-04-23 22:36 EDT`, the corrected reviewer-response seed-`0`
  fixed-`17` coverage jobs are complete with `COMPLETED 0:0`. Corrected
  true-Jacobian/eigendirection job `9347593` wrote
  [results/true_jacobian_geometry_fixed17_seed0_20260423_corrected](/home/mila/l/lia/skae/results/true_jacobian_geometry_fixed17_seed0_20260423_corrected)
  with `49/49` runs, `62,460` rows, `30,014` ok rows, and `0` failures.
  Corrected controlled-transfer root shards `9347590`, `9347591`, and
  `9347592` wrote
  [results/controlled_transfer_switching_fixed17_seed0_20260423_corrected](/home/mila/l/lia/skae/results/controlled_transfer_switching_fixed17_seed0_20260423_corrected)
  with `1,776` rows, `1,632` ok rows, `144` skipped rows, and `0` failures.
  Queue manifests:
  [true queue](/home/mila/l/lia/skae/results/true_jacobian_geometry_fixed17_seed0_20260423_corrected/queue_manifest.json)
  and
  [controlled queue](/home/mila/l/lia/skae/results/controlled_transfer_switching_fixed17_seed0_20260423_corrected/queue_manifest.json).
  Corrected smoke jobs `9347587` and `9347588` completed with
  `COMPLETED 0:0`. The earlier packets
  [results/true_jacobian_geometry_fixed17_seed0_20260423_cached](/home/mila/l/lia/skae/results/true_jacobian_geometry_fixed17_seed0_20260423_cached)
  and
  [results/controlled_transfer_switching_fixed17_seed0_20260423](/home/mila/l/lia/skae/results/controlled_transfer_switching_fixed17_seed0_20260423)
  are retained only as superseded audit artifacts. There is no currently
  running paper-critical SLURM job recorded in this pass.

- Earlier, as of `2026-04-23 20:40 EDT`, no SLURM jobs had yet been submitted
  for the new true-Jacobian/eigendirection or controlled-transfer branches.
  Prepared wrappers:
  [run_transition_rich_true_jacobian_geometry.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_true_jacobian_geometry.sh),
  [queue_transition_rich_true_jacobian_geometry.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_true_jacobian_geometry.sh),
  [run_transition_rich_controlled_transfer_switching.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_controlled_transfer_switching.sh),
  and
  [queue_transition_rich_controlled_transfer_switching.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_controlled_transfer_switching.sh).
  At that point the only validation was shell syntax checking with `bash -n`;
  compute-node smoke validation has since completed.

- As of `2026-04-23 16:43 EDT`, there is no live paper-critical SLURM queue.
  The non-oracle self-routed forecasting packet is complete under
  [results/transition_rich_self_routed_forecasting_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420).
  Queue manifest:
  [self_routed_forecasting_queue.json](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420/automation/self_routed_forecasting_queue.json).
  The original root-only queue `9314170-9314173` was canceled because
  `squeue --start` pushed the shards too far out, and the first reseeded split
  submission on `long-cpu` (`9314196-9314214`) was also canceled after
  scheduler inspection still showed poor start behavior. A temporary
  `main-cpu` workaround (`9314400-9314406`) and a `12h` long-queue retry
  (`9314431-9314437`) were then also canceled once runtime comparisons showed
  that a better backfill shape was available. The final successful queue was
  the one-seed `long-cpu` submission: shards `9314443-9314472` at
  `03:00:00`, with merge `9314473` dependency-held behind them. `sacct`
  now shows every shard and the merge as `COMPLETED 0:0`; shard elapsed times
  ranged from `00:39:51` to `02:08:48`, and the merge finished in `00:00:20`.
  The merged packet writes `510/510` completed runs, `24,600` rows, and
  `0` failures, so the scheduler blocker is fully closed rather than merely
  mitigated.
  The evaluator now also supports resumable intra-shard reruns with atomic
  per-spec flushing, but that patch landed after this queue was submitted, so
  it applies to reruns rather than retroactively to the already-finished
  shards. The resume path itself is compute-validated: validation job
  `9315112` completed in `16s` on `cn-m004` and confirmed that rerunning the
  completed one-spec smoke shard skips work immediately with `1/1` completed.
  Smoke validation remains
  complete under
  [results/transition_rich_self_routed_forecasting_smoke_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_smoke_20260420)
  (`270` rows, `0` failures), and merge-path validation remains complete under
  [results/transition_rich_self_routed_forecasting_merge_smoke_20260420/merged](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_merge_smoke_20260420/merged)
  (`270` rows, `0` failures).

- As of `2026-04-20 15:22 EDT`, the centered-chart mechanism queue is complete under
  [results/transition_rich_centered_chart_mechanism_20260420](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420).
  Queue manifest:
  [centered_chart_mechanism_queue.json](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420/automation/centered_chart_mechanism_queue.json).
  SLURM shards `9310546-9310548` and merge `9310549` all finished
  `COMPLETED 0:0`. The merged packet writes
  [centered_chart_mechanism_rows.csv](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420/centered_chart_mechanism_rows.csv),
  [centered_chart_mechanism_summary.md](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420/centered_chart_mechanism_summary.md),
  and
  [manifest.json](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420/manifest.json)
  with `74,369` rows and `0` failures.

- As of `2026-04-20 14:09 EDT`, there are no paper-critical jobs left in
  `squeue`. The matched hard-init interpretability rerun is complete:
  shards `9304602-9304604` finished, the patched merge reran as `9304747`,
  and the dependent comparison reran as `9304748`. The finalized artifacts are
  [interpretability_final_pass1](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/interpretability_final_pass1)
  and
  [final_comparison_pass1](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1),
  with `13,554` interpretability rows and `0` failures.

- As of `2026-04-19 02:49 EDT`, the matched-hard-init operator-selection
  mechanism packet is also complete under
  [results/transition_rich_operator_selection_hardinit_matched_20260418](/home/mila/l/lia/skae/results/transition_rich_operator_selection_hardinit_matched_20260418).
  Queue manifest:
  [operator_selection_queue.json](/home/mila/l/lia/skae/results/transition_rich_operator_selection_hardinit_matched_20260418/automation/operator_selection_queue.json).
  Smoke job `9304650`, shard jobs `9304655-9304659`, and merge `9304660` all
  finished cleanly. The merged packet writes
  [operator_selection_rows.csv](/home/mila/l/lia/skae/results/transition_rich_operator_selection_hardinit_matched_20260418/operator_selection_rows.csv),
  [operator_selection_summary.md](/home/mila/l/lia/skae/results/transition_rich_operator_selection_hardinit_matched_20260418/operator_selection_summary.md),
  and
  [manifest.json](/home/mila/l/lia/skae/results/transition_rich_operator_selection_hardinit_matched_20260418/manifest.json)
  with `56,538` rows and `0` failures. This queue directly answered the
  review's `support -> local linear law` objection and currently argues
  against the strong version of that claim.

- As of `2026-04-17 17:42 EDT`, the benchmark-aligned Dysts `H30000`
  best-root visual packet is complete. Compute allocation `9295961` on
  `cn-f004` ran
  [tools/generate_dysts_best_root_phase_portraits.py](/home/mila/l/lia/skae/tools/generate_dysts_best_root_phase_portraits.py)
  and wrote
  [dysts_h30000_best_root_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h30000_best_root_phase_portraits_manifest.json)
  plus `15` per-system figure packets under
  [docs/figures/dysts_phase_portraits](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits).
  The packet selects block-diagonal LISTA `sc=6e-3` on `10/15` systems,
  block-diagonal LISTA `sc=3e-3` on `4/15`, and dense LISTA only on
  `dysts:QiChen`. No queue follow-up is required for this artifact set.
- As of `2026-04-17 17:13 EDT`, the older LISTA-only shared-batch Dysts
  `H30000` visual-packet run is also complete. Compute allocation `9295355`
  on `cn-m003` ran
  [tools/generate_dysts_h5000_phase_portraits.py](/home/mila/l/lia/skae/tools/generate_dysts_h5000_phase_portraits.py)
  with `--horizon 30000` and wrote
  [dysts_h30000_lista_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h30000_lista_phase_portraits_manifest.json).
  That packet selects dense LISTA on `14/15` systems and block-diagonal LISTA
  (`sc=6e-3`) only on `dysts:Dadras`; keep it as a historical selector-
  sensitivity artifact rather than the preferred `H30000` appendix packet.
- As of `2026-04-17 09:57 EDT`, the Dysts blockdiag-MLP long-horizon recovery
  chain is complete. Retry `9286093_150` finished in `43m49s`, wrapper
  `9286094` finished, and the chained packet
  `9289755` (`prebuild_dysts_cache`), `9289756` (`dysts_long_eval`
  validation), `9289757` (`dysts_long_eval` array), and `9289758`
  (`dysts_long_collect`) all completed cleanly. The collector under
  [results/dysts_long_horizon_eval_mlp_blockdiag_20260415/collect](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_mlp_blockdiag_20260415/collect)
  reports `300/300` complete tasks and `0` pending tasks.
- As of `2026-04-18 23:24 EDT`, the fixed-`17` matched hard-init MLP control
  packet under
  [results/transition_rich_hardinit_mlp_controls_seed10_20260416](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416),
  launched by `9285895`, has finished its full forecasting-side chain:
  initial array `9285897`, pass-`0` collect / resolve / advance
  `9285898 -> 9285899 -> 9285900`, and rescue pass `1`
  `9291399 -> 9291400 -> 9291401 -> 9291402`.
  [launch_record.env](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/queue_logs/launch_record.env)
  confirms the three hard-init control roots:
  `mlp_sparse_hardinit_basin_partition_control`,
  `mlp_zero_sparse_hardinit_basin_partition_control`, and
  `mlp_sparse_blockdiag_hardinit_basin_partition_control`. Pass-`1`
  [dt_resolution/pass1/dt_resolution.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/dt_resolution/pass1/dt_resolution.md)
  shows all `51/51` arm-system pairs as `accepted_default` with
  `next_request_dt = null`, so no live rescue queue remains for this packet.
  The finalized forecasting summary under
  [collect_pass1/forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/collect_pass1/forecasting_summary.md)
  reports system-median best-periodic `H100/H500/H1000` of
  `0.0082 / 0.0260 / 0.0273` for the sparse hard-init MLP,
  `0.0094 / 0.0359 / 0.0383` for the blockdiag hard-init MLP, and
  `0.5704 / 2.6733 / 3.8044` for the zero-sparse hard-init control. The
  replacement hard-init interpretability chain is also complete now: shard
  reducers `9304602-9304604` finished, the first merge `9304605` failed on an
  empty-cell parsing bug, and the patched reruns `9304747 -> 9304748` wrote
  [interpretability_final_pass1](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/interpretability_final_pass1)
  with `13,554` rows / `0` failures plus the finalized matched-sampling
  comparison in
  [final_comparison_pass1](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1).
  On the paper slice (`absolute:0.001` / `deep`), the two sparse hard-init
  MLP controls are almost tied and the zero-sparse control remains much worse,
  so the queue blocker on this packet is closed and the remaining work is
  paper interpretation rather than execution recovery.
- As of `2026-04-16 13:59 EDT`, no live paper-critical SLURM jobs remain for
  the blockdiag-MLP Dysts extension. The first array `9281462_[0-299%24]`
  failed and should be ignored because of the old `unbound variable` shell
  bug in
  [scripts/run_paper_benchmark_array.sh](/home/mila/l/lia/skae/scripts/run_paper_benchmark_array.sh).
  The replacement queue is finished: wrapper `9282328`, collector `9282331`,
  and compare jobs `9282332-9282334` all completed, while training array
  `9282330_[0-299%24]` finished with `299` `COMPLETED` tasks and `1` `FAILED`
  task. The lone failure is `9282330_150`
  (`generic_sparse_blockdiag_ns200k_sc3em3`, `dysts:Dadras`, seed `0`), whose
  stderr shows a CUDA uncorrectable ECC fault on a Quadro RTX 8000. The
  matching long-horizon wrapper `9282357` is `CANCELLED` with no start time and
  produced no outputs under
  [results/dysts_long_horizon_eval_mlp_blockdiag_20260415](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_mlp_blockdiag_20260415).
  The custom reevaluation root list remains
  [dysts_mlp_blockdiag_long_horizon_roots.tsv](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415/root_specs/dysts_mlp_blockdiag_long_horizon_roots.tsv).
- As of `2026-04-15 20:16 EDT`, the seed-`10` Dysts long-horizon reevaluation
  packet is complete. Launcher `9273653`, cache prebuild `9273655`,
  validation `9273656`, initial collector `9273658`, replacement validation
  probe `9273675`, rescue array `9278881`, and rescue recollect `9278882`
  have all completed. The first main reevaluation array `9273657` had failed
  because older checkpoints serialized legacy config keys such as
  `ENV.COMPETITIVE_LV.SYSTEM_SEED`; that compatibility issue is now fixed in
  [skae/config.py](/home/mila/l/lia/skae/skae/config.py) and the rescue pass
  filled the missing rows. The current collector outputs are
  [forecasting_rows.csv](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/collect/forecasting_rows.csv),
  [pending_rows.csv](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/collect/pending_rows.csv),
  and
  [summary.md](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/collect/summary.md),
  which now report `750/750` complete tasks and `0/750` pending or invalid
  tasks. No live paper-critical SLURM jobs remain for this packet.
- As of `2026-04-14 23:10 EDT`, the non-training fixed-`17` visual-packet run
  under compute allocation `9273591` is complete. It ran
  [tools/generate_fixed17_lista_phase_portraits.py](/home/mila/l/lia/skae/tools/generate_fixed17_lista_phase_portraits.py)
  and wrote
  [fixed17_h1000_h3000_h5000_lista_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/fixed17_lista_phase_portraits_20260414/fixed17_h1000_h3000_h5000_lista_phase_portraits_manifest.json)
  plus `17` per-system selection JSON files and `51` per-horizon phase-
  portrait figure packets. No queue follow-up is required for this artifact
  set.
- As of `2026-04-14 12:17 EDT`, the non-training Dysts visual-packet jobs
  `9269340` and `9269661` are complete. `9269340` ran
  [tools/generate_dysts_h5000_phase_portraits.py](/home/mila/l/lia/skae/tools/generate_dysts_h5000_phase_portraits.py)
  on `main-cpu` and wrote
  [docs/figures/dysts_h5000_lista_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/dysts_h5000_lista_phase_portraits_manifest.json)
  plus `15` per-system H5000 phase-portrait figure packets. `9269661` reran
  the same script at `--horizon 20000` and wrote
  [docs/figures/dysts_h20000_lista_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/dysts_h20000_lista_phase_portraits_manifest.json)
  plus `15` per-system H20000 phase-portrait figure packets. No queue
  follow-up is required for either artifact set.
- As of `2026-04-13 19:16 EDT`, the hard-init seed-`0` chain is complete, the
  clean cross-root compatibility rerun `9211252` is complete, the final
  `200k`, `10`-seed default-`dt` confirmatory packet is fully finalized
  through rescue pass `1`, and the narrow default-sampling `v6` / `v7`
  long-budget floor check has completed as fair default array / collector
  `9228394_[0-339] -> 9228395`. No paper-critical training or collection jobs
  remain live on this branch.
- The April 13 documentation audit found no additional finished paper-critical
  SLURM packets from the last 7 days that were missing from this log. The
  required maintenance was to remove stale wording that still described the
  clean tanh / no-shrink control and several `v7` design-note axes as pending
  after they had already completed.
- Separately, the working-budget zero-sparsity no-shrink control screen under
  [transition_rich_zero_sparse_control_noshrink_20k_seed3_20260410](/home/mila/l/lia/skae/results/transition_rich_zero_sparse_control_noshrink_20k_seed3_20260410)
  finished cleanly as `9223056_[0-50] -> 9223057 -> 9223058`. It collected
  `51` rows, accepted default `dt` on all `17` systems at pass `0`, and
  reported system-median best-periodic `H100/H500/H1000 = 1.8317 / 3.5797 /
  4.1857`.
- The older locked-budget zero-sparsity expansion under
  [transition_rich_zero_sparse_control_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_zero_sparse_control_seed10_20260410)
  as array `9221521_*` is not the requested tanh / no-shrink control because
  its task table maps `mlp_zero_sparse_basin_partition_control` to
  `generic_sparse` rather than `generic_no_shrink`. It should therefore not be
  used as the clean anti-sparsity packet. It is still scientifically valuable
  as a locked-budget ReLU-only ablation, because it removes the explicit
  penalty while keeping the encoder's architectural shrinkage. Operationally
  it now runs through collect / resolve pass `4`, still accepts default `dt`
  on all `17` systems, and remains at system-median best-periodic
  `H100/H500/H1000 = 0.5764 / 2.0556 / 2.6532` with `17/17` good systems.
- The corrected locked-budget tanh / no-shrink control is now complete under
  [transition_rich_zero_sparse_tanh_control_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_zero_sparse_tanh_control_seed10_20260410)
  as wrapper `9224111`. That launcher completed successfully and emitted a
  fresh default task table with `config_name=generic_no_shrink`. The clean
  default pass now completes as `9224263_* -> 9224264 -> 9224265`; despite one
  fast array failure, the collector writes `169` rows, all `17` systems accept
  default `dt`, and the packet reports system-median best-periodic
  `H100/H500/H1000 = 0.5763 / 1.7924 / 2.4279`. This packet uses
  `MAX_HALVINGS=0`, so no rescue pass was needed.
- A new narrow LISTA-only seed-`0`, `20k` follow-up is now complete under
  [transition_rich_basin_partition_20260410_seed0_smoke_v7](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260410_seed0_smoke_v7)
  as wrapper `9226563`. It reuses the earlier `20k` shortlist anchors and
  launches only new unrun LISTA variants:
  `lista_blockdiag_adaptive_groupwise_threshold_basin_partition`,
  `lista_blockdiag_sparsegroup_basin_partition`,
  `lista_blockdiag_signsplit_momentum_basin_partition`,
  `lista_dense_softblock_dict_tied_precode_basin_partition`,
  `lista_dense_softblock_hybrid_precode_basin_partition`,
  `lista_dense_softblock_signsplit_p64_softblock5em4_basin_partition`,
  `lista_dense_softblock_signsplit_p64_softblock1em3_basin_partition`, and
  `lista_dense_softblock_signsplit_p64_momentum_basin_partition`. The wrapper
  completed and emitted `136` default-`dt` tasks (`8` variants x `17` systems
  x `1` seed). The chain `9226564_[0-135] -> 9226565 -> 9226566 -> 9226567`
  is now complete with `0` reducer failures. Its task table leaves
  `hard_init_oversample` unset, so this wave stays on the default sampling
  regime and keeps sampling out of the causal comparison against the MLP
  controls. The best new forecasting root is
  `lista_blockdiag_sparsegroup_basin_partition`
  (`H1000 system-median best = 0.0846`), while the strongest new deep-basin
  support-compression read is
  `lista_dense_softblock_signsplit_p64_softblock5em4_basin_partition`
  (`H(S|B)=0.6795`, `U_exact=0.8453`, `H(F|B)=0.0634` at
  `absolute:0.001` / `deep`), but that root is missing `1/17` systems after a
  fast training failure. No new `v7` root cleanly dominates both forecasting
  and support alignment, so keep the completed hard-init packet as the main
  shortlist provenance and treat `v7` as supporting method-side evidence.
- A narrow default-sampling `200k`, `10`-seed forecasting follow-up is now
  complete under
  [transition_rich_basin_partition_v6_v7_200k_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_basin_partition_v6_v7_200k_seed10_20260410)
  as wrapper `9228393`. It promotes the best forecasting roots from `v6` and
  `v7`,
  `lista_dense_softblock_signsplit_coherence_basin_partition` and
  `lista_blockdiag_sparsegroup_basin_partition`, to `200k` with `10` seeds on
  the same default sampling regime. The wrapper wrote a `340`-task default
  table (`17` systems x `2` roots x `10` seeds) and then attempted to queue a
  model-specific dt-rescue continuation, but that would have broken the
  matched-dt comparison because rescue is decided per model/system arm. Those
  rescue-dependent jobs `9228396-9228399` were therefore canceled, and the
  wrapper exit is not the scientific result. The actual fair queue,
  `9228394_[0-339] -> 9228395`, completed on April 11, 2026 and wrote
  `329/340` forecasting rows across all `17` systems. The better coherence
  root reports system-median best-periodic `H100/H500/H1000 = 0.0416 / 0.0761 / 0.0796`;
  the sparsegroup root reports `0.0437 / 0.1142 / 0.1193`. This is not enough
  to reopen the shortlist, and because the packet is forecasting-only it adds
  no new basin-support reduction.
- If that narrow `v6` / `v7` long-budget follow-up does not materially close
  the forecasting gap to the promoted `v5` lineage, the next paper-facing
  diagnostic should still be the evaluation pass on the locked finalists and
  controls that reports `H100`, `H500`, and `H1000` by basin depth /
  separatrix proximity rather than another broad training sweep. The mechanism
  target is explicit: if sparsity helps because the latent tracks basin
  identity, long-horizon forecasting should be best deep in a basin and worst
  near a separatrix.
- The completed method-side chain is the seed-`0` hard-init follow-up under
  [transition_rich_basin_partition_hardinit_seed0_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409).
  Chain `9209614` (training array) -> `9209615` (`collect_tr_bp`) ->
  `9209616` (`tr_interp_reduce`) completed cleanly.
- That hard-init packet contains `68` training tasks:
  `17` fixed-shortlist systems x `4` roots x `1` seed at the standard `20k`
  budget. Its role is narrow and method-side: test whether oversampling hard
  initial conditions near separatrices improves the current `v5` sign-split
  block-diagonal and dense-soft-block `p64` roots enough to justify a larger
  rerun.
- The queued roots in that packet are
  `lista_blockdiag_signsplit_basin_partition`,
  `lista_blockdiag_signsplit_hardinit_basin_partition`,
  `lista_dense_softblock_signsplit_p64_basin_partition`, and
  `lista_dense_softblock_signsplit_p64_hardinit_basin_partition`.
- A prepared but not currently queued `3`-seed expansion of the same follow-up
  already exists under
  [transition_rich_basin_partition_hardinit_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_20260409).
  Its task manifest contains `204` runs (`17 x 4 x 3`). Do not launch it by
  default; use the live seed-`0` reduction to decide whether the signal is
  strong enough to warrant a paper-facing robustness confirmation.
- The first post-hard-init cross-root phase-`1` comparison has already been
  launched under
  [transition_rich_post_hardinit_crossroot_eval_20260409](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409).
  Its dependency chain was `9210427` (`collect_tr_crossroot`,
  `afterok:9209616`) -> `9210429` (`tr_interp_crossroot`,
  `afterok:9210427`). Both jobs completed.
- That queued bundle is defined in
  [selected_roots.txt](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409/root_specs/selected_roots.txt)
  with unique labels so the reduction can distinguish the original `v5`
  shortlist winners from the hard-init packet's retrained base roots. The
  bundle covers `v5` forecast-retaining and exact-support leaders, the
  strongest `v6` coherence and restrained-linear follow-ups, the matched `v1`
  sparse MLP control, and all four hard-init packet roots.
- Rationale for that queued comparison:
  it is the minimal paper-facing bundle that can answer the main remaining
  causal question without spending more training budget first. It puts the
  strongest completed LISTA roots, the main `v6` identifiability follow-ups,
  the matched non-LISTA control, and the near-separatrix oversampling ablation
  under one common reduction protocol on the fixed `17` systems.
- That first submitted phase-`1` reduction is now superseded operationally by
  the clean rerun
  [interpretability_rerun_compat_20260409](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409/interpretability_rerun_compat_20260409),
  job `9211252`. That rerun completed cleanly with `4131` interpretability
  rows and `0` failures, so the seed-`0` paper-facing ranking is now fixed:
  `hardinit_packet_blockdiag_hardinit` is the forecast-retaining
  interpretability finalist, `hardinit_packet_dense_p64_hardinit` is the
  stronger forecasting / intervention tradeoff, and the matched sparse MLP
  control does not displace either finalist on the selected deep-basin
  `absolute:0.001` slice.
- The final confirmatory packet under
  [transition_rich_basin_partition_final_seed10_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409)
  currently includes `680` default-source tasks:
  `17` systems x `4` roots x `10` seeds at `200k`, comparing
  `lista_blockdiag_signsplit_hardinit_basin_partition`,
  `lista_dense_softblock_signsplit_p64_hardinit_basin_partition`,
  `mlp_sparse_basin_partition_control`, and the newly exposed
  `mlp_zero_sparse_basin_partition_control`.
- The confirmatory array's small hardware-only failure cluster on `cn-a009`
  (tasks `238`, `255`, `267`, `279`, `295`, and `303`) was resolved by rescue
  pass `1`; no additional rescue rows were requested after
  `9214919`.
- The finalized forecasting summary is now materialized at
  [collect_pass1/forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/collect_pass1/forecasting_summary.md).
  On system-median best-periodic `H1000`, it reports `0.0516` for the
  block-diagonal hard-init LISTA finalist, `0.0775` for the dense soft-block
  `p=64` hard-init LISTA finalist, `0.0608` for the matched sparse MLP
  control, and `0.0909` for the zero-`L1` ReLU MLP ablation, with all
  four roots at `17/17` good systems.
- The corresponding locked-packet state-level comparison is now materialized
  under
  [interpretability_final_pass1/interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/interpretability_final_pass1/interpretability_summary.md)
  and
  [final_comparison_pass1/transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md).
  On the selected deep `absolute:0.001` slice, the dense `p64` hard-init
  LISTA root is the only clear basin-support winner over the matched sparse
  MLP control, while the block-diagonal hard-init root remains the stronger
  forecast-retaining companion rather than the lead exact-support result.
- Mila's `AssocMaxSubmitJobLimit` rejects a fully pre-expanded multi-pass
  rescue chain at this scale, so the live seed-`10` packet is now following an
  incremental queueing pattern: default array `9211290_[0-679]`, collect
  `9211291`, resolve `9211292`, then submit only the rescue passes that the
  resolver actually requests. A dedicated one-pass launcher now exists at
  [queue_transition_rich_basin_partition_rescue_pass.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_basin_partition_rescue_pass.sh)
  so later rescue passes can be queued without pre-expanding the whole chain.
- That incremental path is now also connected to a watcher that has already
  fired:
  [advance_transition_rich_basin_partition_packet.sh](/home/mila/l/lia/skae/scripts/advance_transition_rich_basin_partition_packet.sh)
  completed as `9211747` after dependency `afterany:9211292` and emitted the
  actual rescue pass-`1` array `9214917_*`. The same path then finalized the
  packet automatically at pass `1`; see
  [advance_pass1.json](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/automation/advance_pass1.json)
  for the recorded reducer / summary job ids and finalized status.
- That queued post-hard-init chain is intentionally phase `1` only:
  it runs with the existing reducer on raw support-family / projection /
  operator-family / Jacobian-family metrics. Phase `2` remains queued only on
  paper, not in SLURM, until the reducer is extended with the April 9
  basis-aware alignment metrics.
- Historical note on the first submitted cross-root phase-`1` reduction:
  `9210429` wrote `0` interpretability rows and `17` failures under
  [failures.json](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409/interpretability/failures.json)
  because `ROOT_LABELS_CSV` arrived as only `v5_blockdiag_signsplit` and the
  saved block-diagonal LISTA checkpoints exposed `encoder.We.*` keys instead
  of the newer `precode_module` / `dict_param` layout. That invalid first
  attempt is now only provenance; it was superseded by the clean rerun
  `9211252` documented above.
- `v1` closed cleanly under
  [transition_rich_basin_partition_20260409_seed0_smoke_v1](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v1).
  Chain `9202665 -> 9202666_[0-84] -> 9202667 -> 9202673 -> 9202674`
  completed. This packet contains the matched sparse MLP control, so the
  causal control data now exist locally even though the paper-facing
  comparison is not yet written up.
- `v2` closed cleanly under
  [transition_rich_basin_partition_20260409_seed0_smoke_v2](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v2).
  Chain `9202733 -> 9202734_[0-67] -> 9202735 -> 9202741 -> 9202742`
  completed.
- `v3` closed cleanly under
  [transition_rich_basin_partition_20260409_seed0_smoke_v3](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v3).
  Chain `9202783 -> 9202785_[0-50] -> 9202786 -> 9202804 -> 9202805 ->
  9202814/9202815` completed.
- `v4` is closed at its emitted scope under
  [transition_rich_basin_partition_20260409_seed0_smoke_v4](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v4).
  Launcher `9202903` still failed with `AssocMaxSubmitJobLimit`, but every
  emitted default / rescue / reduce job (`9202904`, `9202905`, `9202906`,
  `9202907`, `9202908`, `9202909`, `9202910`, `9202911`) completed
  successfully.
- `v5` closed cleanly under
  [transition_rich_basin_partition_20260409_seed0_smoke_v5](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v5).
  Chain `9203114 -> 9203116 -> 9203117 -> 9203118` completed. Forecasting
  summary:
  [forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v5/collect_pass0/forecasting_summary.md).
  Interpretability summary:
  [interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v5/interpretability_pass0/interpretability_summary.md).
  Current shortlist winner for exact-support compression:
  `lista_dense_softblock_signsplit_p64_basin_partition`. Current shortlist
  winner for forecast retention:
  `lista_blockdiag_signsplit_basin_partition`.
- `v6` closed cleanly under
  [transition_rich_basin_partition_20260409_seed0_smoke_v6](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v6).
  Chain `9203315 -> 9203316 -> 9203317 -> 9203318` completed. Forecasting
  summary:
  [forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v6/collect_pass0/forecasting_summary.md).
  Interpretability summary:
  [interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260409_seed0_smoke_v6/interpretability_reduce_pass0/interpretability_summary.md).
  Packet read: coherence improves forecasting, linear encoders improve
  support compression, but no `v6` root displaces `v5 p64` on the overall
  frontier.
- The historical fixed-shortlist LISTA reduction remains complete under
  [basin_support_metrics_20260408_v3](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/basin_support_metrics_20260408_v3),
  and the native-trio study-plan smoke remains complete under
  [transition_rich_interpretability_smoke_20260409/native_seed0_v2_family_projection](/home/mila/l/lia/skae/results/transition_rich_interpretability_smoke_20260409/native_seed0_v2_family_projection).
- There is no dependency backlog blocking interpretation now. The next manual
  branch decision comes after the cross-root phase-`1` reduction is rerun
  cleanly. At that point the live choice is whether any hard-init variant
  actually moves the paper-facing rank order enough to justify a narrow
  `3`-seed expansion, or whether the branch should stop training-side
  queueing and finish the phase-`2` basis-aware reducer extension plus rerun.
- Forward training-budget policy for interpretability runs:
  all new queues on the fixed `17`-system shortlist should use `20000` steps,
  and `200000` should be reserved for the final locked paper rerun only.
- The active forward experimental scope is now frozen to `17` systems:
  `multiwell_strong_transition`, `gated_local_linear`,
  `gated_transfer_linear`, `arrested_spiral`, `cal_asymmetric_3`,
  `cal_high_cross_3`, `cal_hexagon_6`, `cal_octagon_8`, `cal_pentagon_5`,
  `cal_square_4`, `checkerboard_potential`, `duffing_triple_well`,
  `snic_multi`, `transition_routes_4`, `var_depth_gradient_4`,
  `var_diamond_4`, and `var_l_shape_5`.
- Default-`dt` sweep status:
  - launcher `9190857`: completed `0:0`
  - first array `9190869`: `16/34` tasks completed directly and `18/34` tasks
    failed and were rerouted
  - rerun array `9192341`: the `18` rerouted tasks all completed `0:0`
  - scratch result root:
    [transition_rich_basin_partition_20260407](/network/scratch/l/lia/skae/transition_rich_basin_partition_20260407)
- `dt`-rescue chain status:
  - launcher `9193402`: completed `0:0`
  - dependent collector/resolve chain `9193424-9193443`: completed `0:0`
  - dependent rescue arrays `9193426`, `9193429`, `9193432`, `9193435`,
    `9193438`, `9193441`: all completed their submitted work
  - the rescue task tables themselves were already rewritten to `20k`, but the
    completed rescue waves should be read as repeated default-`dt` reruns
    rather than as true halved-`dt` evidence because the collector had dropped
    `env_dt` on `gated_*` and `claude:*` rows
  - do not queue pass `7` from these artifacts; the scientific step-size
    question is already answered for this one-seed LISTA packet
- Natural live-sweep rollout artifacts are not available yet:
  no `evaluation_best/.../rollout_artifacts.pt` files have appeared under the
  live result roots so far.
- Manual native-system diagnostics already exist under:
  [manual_eval](/network/scratch/l/lia/skae/transition_rich_basin_partition_20260407/manual_eval)
- One new local catalog-audit pass is complete without changing the live queue:
  - audit note:
    [docs/planning/claude_catalog_audit_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_audit_20260407.md)
  - audit figure root:
    [docs/figures/claude_catalog_audit_20260407](/home/mila/l/lia/skae/docs/figures/claude_catalog_audit_20260407)
  - key result:
    the combined grounded Claude-catalog fast screen now supports `12`
    accepted systems across `83` screened systems, with an `8`-system
    strict-crossing core, not the stale `44 confirmed passing` headline in the
    older branch note
- No Claude-catalog training packet has been launched yet. The existing
  `claude_catalog_packet` manifest/task-builder/queue launcher now follow the
  same `20k` diagnostic budget, but they still encode the superseded
  `6`-system recommendation and should be treated as historical scaffolding
  rather than the active scope definition:
  - historical handoff / packet note:
    [docs/planning/claude_catalog_handoff_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_handoff_20260407.md)
  - historical manifest:
    [claude_catalog_packet_manifest.py](/home/mila/l/lia/skae/skae/benchmarks/claude_catalog_packet_manifest.py)
  - historical task builder:
    [build_claude_catalog_packet_tasks.py](/home/mila/l/lia/skae/tools/build_claude_catalog_packet_tasks.py)
  - historical compute-node queue launcher:
    [queue_claude_catalog_packet.sh](/home/mila/l/lia/skae/scripts/queue_claude_catalog_packet.sh)
- The completed transition-rich execution chain is:
  - Stage 1 array `9135303`: finished with `26/27` completed tasks and one
    failed cell, `9135303_20 = lista_dense_promoted_stage4 x multiwell_strong_transition x seed_2`
  - transition-rich collect job `9135304`: completed `0:0`
  - pairwise compare jobs `9135305`, `9135306`, `9135307`: completed `0:0`
  - post hoc chart-change attribution job `9135358`: completed `0:0`
  - post hoc support local-linearity job `9135411`: completed `0:0`
  - repo-side Stage 1 root:
    [results/transition_rich_screening_stage1_20260401](/home/mila/l/lia/skae/results/transition_rich_screening_stage1_20260401)
  - repo-side chart-attribution root:
    [results/transition_rich_chart_change_attribution_20260401](/home/mila/l/lia/skae/results/transition_rich_chart_change_attribution_20260401)
  - repo-side support-local-linearity root:
    [results/transition_rich_support_local_linearity_20260401](/home/mila/l/lia/skae/results/transition_rich_support_local_linearity_20260401)
  - scratch run root: `/network/scratch/l/lia/skae/transition_rich_screening_stage1_20260401`
- One new local benchmark-validity experiment is complete without changing the
  live queue:
  - oracle chart-switch collector root:
    [results/transition_rich_oracle_chart_switch_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_chart_switch_20260401)
  - execution/result note:
    [docs/planning/transition_rich_oracle_chart_switch_benchmark_plan_20260401.md](/home/mila/l/lia/skae/docs/planning/transition_rich_oracle_chart_switch_benchmark_plan_20260401.md)
- A second local benchmark-validity experiment is also complete without
  changing the live queue:
  - oracle refresh-cadence collector root:
    [results/transition_rich_oracle_refresh_cadence_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_refresh_cadence_20260401)
  - execution/result note:
    [docs/planning/transition_rich_oracle_refresh_cadence_plan_20260401.md](/home/mila/l/lia/skae/docs/planning/transition_rich_oracle_refresh_cadence_plan_20260401.md)
- The completed matrix was the full seed-robust Stage 1 screen, not a one-seed
  pilot:
  - `multiwell_strong_transition`, `gated_local_linear`,
    `gated_transfer_linear`
  - `generic_sparse_ns200k_best`, `generic_sparse_sc0_ns200k_best`,
    `lista_dense_promoted_stage4`
  - seeds `0,1,2`
- The March 25 seed-10 robustness backfill and the March 30 verification pass closed the paper-facing forecasting seed-coverage gaps used by `docs/review_main_results_tables_20260314.tex`.
- The March 31 raw-source seed-statistics reanalysis is now written in `docs/PAPER_SEED_STATISTICS_20260331.md`; it verifies raw-vs-collector agreement and records the remaining raw finite-value coverage gaps explicitly.
- The March 21 matched zero-sparsity MLP campaign is fully collected across benchmark, hard-system, and mechanism families.
- The March 17-20 repaired block-`K` fairness-control wave is fully collected and replaces the invalid historical MLP `+ block_diagonal K` rows.
- The tests-first transition-rich implementation pass is now complete locally:
  - the new tests pass
  - the new calibration tool exists at [tools/calibrate_transition_system.py](/home/mila/l/lia/skae/tools/calibrate_transition_system.py)
  - two secondary `2D` candidates are frozen from deterministic calibration
  - the explicit-transfer toy `gated_transfer_linear` is now
    implemented, calibrated, and plotted under
    [docs/figures/chart_switching_transfer_20260331](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331)
- The standardized evaluation stack now also contains a deterministic flow-branching diagnostic for the frozen `2D` systems, with threshold sweeps, ground-truth null baselines, and `2D` visual artifacts written through the usual checkpoint evaluation path.
- The new transition-rich screening tooling now also exists and passes local
  tests:
  - task builder:
    [tools/build_transition_rich_screening_tasks.py](/home/mila/l/lia/skae/tools/build_transition_rich_screening_tasks.py)
  - screening summary collector:
    [tools/summarize_transition_rich_screening.py](/home/mila/l/lia/skae/tools/summarize_transition_rich_screening.py)
  - SLURM wrappers:
    [scripts/collect_transition_rich_screening.sh](/home/mila/l/lia/skae/scripts/collect_transition_rich_screening.sh),
    [scripts/queue_transition_rich_screening_stage1_20260401.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_screening_stage1_20260401.sh)
  - focused validation:
    `uv run python -m pytest tests/test_transition_rich_screening_tasks.py tests/test_transition_rich_screening_summary.py tests/test_paper_followup_recipe_tasks.py -q` -> `4 passed`
- The post hoc chart-change attribution tooling for `gated_transfer_linear`
  now also exists and passes local tests:
  - plan:
    [docs/planning/transition_rich_chart_change_attribution_plan_20260401.md](/home/mila/l/lia/skae/docs/planning/transition_rich_chart_change_attribution_plan_20260401.md)
  - per-run evaluator:
    [tools/evaluate_transition_rich_chart_change_attribution.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_chart_change_attribution.py)
  - batch collector:
    [tools/collect_transition_rich_chart_change_attribution.py](/home/mila/l/lia/skae/tools/collect_transition_rich_chart_change_attribution.py)
  - SLURM wrappers:
    [scripts/collect_transition_rich_chart_change_attribution.sh](/home/mila/l/lia/skae/scripts/collect_transition_rich_chart_change_attribution.sh),
    [scripts/queue_transition_rich_chart_change_attribution_20260401.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_chart_change_attribution_20260401.sh)
  - focused validation:
    `uv run python -m pytest tests/test_transition_rich_chart_change_attribution.py tests/test_transition_rich_screening_tasks.py tests/test_transition_rich_screening_summary.py -q` -> `7 passed`
- The post hoc support local-linearity tooling for the transition-rich suite
  now also exists and passes local tests:
  - plan:
    [docs/planning/transition_rich_support_local_linearity_plan_20260401.md](/home/mila/l/lia/skae/docs/planning/transition_rich_support_local_linearity_plan_20260401.md)
  - per-run evaluator:
    [tools/evaluate_transition_rich_support_local_linearity.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_support_local_linearity.py)
  - batch collector:
    [tools/collect_transition_rich_support_local_linearity.py](/home/mila/l/lia/skae/tools/collect_transition_rich_support_local_linearity.py)
  - SLURM wrappers:
    [scripts/collect_transition_rich_support_local_linearity.sh](/home/mila/l/lia/skae/scripts/collect_transition_rich_support_local_linearity.sh),
    [scripts/queue_transition_rich_support_local_linearity_20260401.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_support_local_linearity_20260401.sh)
  - focused validation:
    `uv run python -m pytest tests/test_transition_rich_support_local_linearity.py tests/test_transition_rich_chart_change_attribution.py tests/test_transition_rich_screening_tasks.py tests/test_transition_rich_screening_summary.py -q` -> `10 passed`
- The explicit-transfer toy now also passes real training-entry smokes through the
  actual CLI:
  - full generic-sparse smoke with standardized evaluation:
    [runs/transition_rich_smoke/generic_sparse_gated_transfer/20260401-003500/20260401-003125](/home/mila/l/lia/skae/runs/transition_rich_smoke/generic_sparse_gated_transfer/20260401-003500/20260401-003125)
  - zero-sparsity MLP smoke:
    [runs/transition_rich_smoke/zero_sparse_gated_transfer/20260401-010000/20260401-004017](/home/mila/l/lia/skae/runs/transition_rich_smoke/zero_sparse_gated_transfer/20260401-010000/20260401-004017)
  - dense LISTA smoke:
    [runs/transition_rich_smoke/lista_dense_gated_transfer/20260401-010000/20260401-004017](/home/mila/l/lia/skae/runs/transition_rich_smoke/lista_dense_gated_transfer/20260401-010000/20260401-004017)
- A real-artifact validation of the new summary collector also exists on the
  generic-sparse transfer smoke under
  [results/transition_rich_smoke_gated_transfer_summary_20260401](/home/mila/l/lia/skae/results/transition_rich_smoke_gated_transfer_summary_20260401).
- A real-artifact validation of the chart-change attribution collector also
  exists on the same undertrained generic-sparse transfer smoke under
  [results/transition_rich_smoke_chart_change_attr_20260401](/home/mila/l/lia/skae/results/transition_rich_smoke_chart_change_attr_20260401).
- A real-artifact validation of the support local-linearity stage now also
  exists on the undertrained transfer smokes under
  [results/transition_rich_smoke_support_local_linearity_20260401](/home/mila/l/lia/skae/results/transition_rich_smoke_support_local_linearity_20260401),
  with collector summary in
  [collector/summary.md](/home/mila/l/lia/skae/results/transition_rich_smoke_support_local_linearity_20260401/collector/summary.md)
  and per-run outputs for `generic_sparse`, zero-sparsity MLP, and dense LISTA
  in
  [per_run](/home/mila/l/lia/skae/results/transition_rich_smoke_support_local_linearity_20260401/per_run).
- Do not queue another broad benchmark or hard-system rerun by default while
  the transition-rich Stage 1 screen is live.

## April 1, 2026: Oracle Chart-Switch Benchmark Validity Read

Concrete result(s):
- The local oracle benchmark is complete under
  [results/transition_rich_oracle_chart_switch_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_chart_switch_20260401).
- At `H20` on all starts, oracle chart switching beats a single global linear
  fit by:
  - `0.129974` on `gated_local_linear`
  - `0.0206368` on `gated_transfer_linear`
  - `0.0144757` on `multiwell_strong_transition`
- At `H20` on all starts, oracle chart switching beats fixed-chart local
  rollouts by:
  - `0.0219174` on `gated_local_linear`
  - `0.00237432` on `gated_transfer_linear`
  - `0.00539429` on `multiwell_strong_transition`
- The harder-subset read is the key discriminator:
  - `gated_local_linear` chart-switch starts: `+0.157948` vs global,
    `+0.0293535` vs fixed-chart
  - `multiwell_strong_transition` chart-switch starts: `+0.0593379` vs global,
    `+0.0245243` vs fixed-chart
  - `gated_transfer_linear` chart-switch starts: `+0.0208039` vs global,
    `+0.00258307` vs fixed-chart
  - `gated_transfer_linear` transfer starts: `-0.0557357` vs global,
    `+0.00993001` vs fixed-chart
  - `gated_transfer_linear` transition-region starts: `-0.0118927` vs global,
    `+0.0174968` vs fixed-chart

Result in experimental context:
- This was a ground-truth benchmark-validity experiment, not a learned-model
  benchmark.
- We fit affine state-space maps directly to held-out transitions from the true
  deterministic systems and compared:
  - one global linear map
  - one fixed local chart per rollout
  - oracle chart switching under the true chart schedule
  - shuffled chart switching with matched chart counts
- The plan and exact evaluator live in
  [docs/planning/transition_rich_oracle_chart_switch_benchmark_plan_20260401.md](/home/mila/l/lia/skae/docs/planning/transition_rich_oracle_chart_switch_benchmark_plan_20260401.md),
  [tools/evaluate_transition_rich_oracle_chart_switch.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_oracle_chart_switch.py),
  and
  [tools/collect_transition_rich_oracle_chart_switch.py](/home/mila/l/lia/skae/tools/collect_transition_rich_oracle_chart_switch.py).

Interpretation:
- `gated_local_linear` is the cleanest mechanistic chart-switching positive in
  the suite.
- `multiwell_strong_transition` is a weaker but real secondary positive, with
  gains concentrated on switching trajectories.
- `gated_transfer_linear` remains scientifically useful because the true chart
  partition matters and fixed-chart local models are worse than oracle
  switching, but it is not the cleanest proof that chart switching dominates:
  on the hardest transfer and transition-region starts, a single global linear
  fit is still competitive or better.

Project implications:
- The suite should stay three-system, but paper positioning must change.
- `gated_local_linear` should carry the clean main-text mechanistic claim.
- `gated_transfer_linear` should be written as the explicit-transfer stress test
  rather than as the only flagship proof of chart-switching linearizations.
- `multiwell_strong_transition` should remain in the paper as a weaker
  shared-corridor secondary toy.

Next steps:
- Compare the collected trained-model results directly against this oracle
  ranking.
- If `gated_transfer_linear` stays weak relative to the oracle benchmark after
  the collected model read, retune or demote it only if the paper still needs a
  stronger transfer-localization benchmark.

## April 1, 2026: Oracle Refresh-Cadence Benchmark Validity Read

Concrete result(s):
- The local oracle refresh-cadence benchmark is complete under
  [results/transition_rich_oracle_refresh_cadence_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_refresh_cadence_20260401).
- At `H20`, the recovered fraction of the oracle-vs-fixed improvement on all
  starts is:
  - `gated_local_linear`: `0.988651` at `c=2`, `0.90577` at `c=5`,
    `0.665843` at `c=10`
  - `multiwell_strong_transition`: `0.968223` at `c=2`, `0.858159` at `c=5`,
    `0.639087` at `c=10`
  - `gated_transfer_linear`: `0.88425` at `c=2`, `0.503731` at `c=5`,
    `0.0577701` at `c=10`
- The largest cadence still preserving at least `90%` of the oracle-vs-fixed
  gain on all starts is:
  - `gated_local_linear`: `c=5`
  - `multiwell_strong_transition`: `c=2`
  - `gated_transfer_linear`: only `c=1`
- The true held-out median chart dwell lengths are:
  - `gated_local_linear`: `171`
  - `multiwell_strong_transition`: `92`
  - `gated_transfer_linear`: `33`
- On the hardest `gated_transfer_linear` transfer starts, oracle periodic
  refresh still improves over stale fixed local charts but not over the global
  fit:
  - `c=2`: `gain_vs_fixed = +0.00866824`, `gain_vs_global = -0.0569974`
  - `c=5`: `gain_vs_fixed = +0.00483172`, `gain_vs_global = -0.0608339`

Result in experimental context:
- This was a ground-truth benchmark-validity experiment, not a learned-model
  benchmark.
- We fit affine state-space local maps on held-out trajectories from the true
  systems, then refreshed the true chart identity every `k` steps while holding
  it fixed between refreshes.
- The causal question was whether the benchmark itself supports a periodic
  re-encoding interpretation, not just the extreme endpoints of “never refresh”
  versus “refresh every step.”

Interpretation:
- `gated_local_linear` is the cleanest periodic-refresh benchmark in the suite.
  It preserves most of the oracle benefit even at moderate cadence and degrades
  smoothly as chart identity becomes stale.
- `multiwell_strong_transition` is still a real positive, but it needs faster
  refresh than `gated_local_linear`.
- `gated_transfer_linear` is much more cadence-sensitive and its relevant
  comparison is stale local charts, not global linear dominance. That makes it
  a strong stress test, not the cleanest mechanistic benchmark.

Project implications:
- The paper now has two independent ground-truth reads supporting the same role
  split:
  - `gated_local_linear` should carry the clean main-text mechanistic claim
  - `multiwell_strong_transition` should remain the weaker secondary positive
  - `gated_transfer_linear` should be written as the explicit-transfer stress
    test
- This also sharpens the trained-model decision rule: learned periodic gains
  should be compared against the oracle cadence ranking, not only against the
  oracle per-step chart-switch ranking.

Next steps:
- Use the collected Stage 1 and post hoc reads below as the active paper-facing
  evidence for the transition-rich branch.
- Treat `gated_local_linear` as the main-text mechanistic positive and
  `gated_transfer_linear` as the transfer stress test unless a new benchmark
  overtakes them.
- Do not claim chart-change localization on `gated_transfer_linear` from the
  current evidence.

## April 1, 2026: Collected Stage 1 Transition-Rich Screen

Concrete result(s):
- The full `200k` three-system Stage 1 screen is collected under
  [results/transition_rich_screening_stage1_20260401](/home/mila/l/lia/skae/results/transition_rich_screening_stage1_20260401).
- `26/27` runs completed. The only missing cell is
  `lista_dense_promoted_stage4` on `multiwell_strong_transition`, seed `2`
  (`9135303_20` failed).
- `gated_local_linear` is a clean positive for all three model roots:
  - `H1000` median best-periodic: `0.0890151` (`generic_sparse`),
    `0.000710212` (zero-sparsity MLP), `0.000783794` (dense LISTA)
  - `H1000` median no-reencode: `27651.5`, `2.86748`, `206.989`
- `gated_transfer_linear` is also a clean positive for all three model roots:
  - `H1000` median best-periodic: `1.7954`, `1.80077`, `1.09863`
  - `H1000` median no-reencode: `6.54401e+20`, `2.45499e+34`, `1.25639e+12`
- `multiwell_strong_transition` is not a clean forecasting positive:
  - `H100` median best-periodic stays very large across all three roots:
    `310.41`, `94.5872`, `180.29`
  - `H500` median best-periodic is catastrophically unstable:
    `3.12343e+20`, `2.70981e+13`, `3.05402e+19`
  - only isolated `H1000` medians become small on some roots/seeds, so the
    system does not provide a stable paper-facing positive at the current
    budget
- The flow-branching reference rate under the selected periodic evaluation also
  separates the suite:
  - `gated_transfer_linear`: near zero (`0`, `0`, `6.127e-05`)
  - `gated_local_linear`: low but nonzero (`0.0199912`, `0.02078`, `0.0631283`)
  - `multiwell_strong_transition`: much higher (`0.233598`, `0.179023`,
    `0.306753`)

Result in experimental context:
- This was the first seed-robust learned-model screen on the calibrated
  three-system suite:
  `3` systems x `3` model roots x `3` seeds at the paper `200k` budget.
- The official forecasting summary is
  [forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_screening_stage1_20260401/collect/forecasting/forecasting_summary.md),
  and the branch-specific transition-rich summary is
  [transition_rich_screening_summary.md](/home/mila/l/lia/skae/results/transition_rich_screening_stage1_20260401/collect/transition_rich/transition_rich_screening_summary.md).
- The Stage 1 acceptance rule was never “win on every toy”; it was to find out
  which systems become clean positives, which remain hard, and whether periodic
  re-encoding matters at all on the chart-switching suite.

Interpretation:
- The learned-model read agrees with the oracle role split on the two strongest
  systems:
  `gated_local_linear` is the clean mechanistic positive and
  `gated_transfer_linear` is a strong transfer-oriented forecasting positive.
- The strongest robust empirical effect in the branch is periodic re-encoding
  versus stale latent rollouts, not explicit sparsity versus the matched
  zero-sparsity MLP.
- `multiwell_strong_transition` remains scientifically interesting, but it is
  too unstable to serve as a clean flagship positive at the current budget.

Project implications:
- The transition-rich branch is now model-backed rather than oracle-only.
- Main-text benchmark positioning should center:
  - `gated_local_linear` as the clean mechanistic positive
  - `gated_transfer_linear` as the hard explicit-transfer stress test
- `multiwell_strong_transition` should be written as a weaker secondary toy or
  demoted if we do not invest in a targeted follow-up.
- The zero-sparsity MLP is competitive enough that the branch should currently
  be written as a periodic re-encoding and local-partition story, not as a
  sparse-only win.

Next steps:
- Do not queue another broad screen by default.
- Decide whether the missing dense-LISTA `multiwell_strong_transition` seed is
  worth rerunning only after `multiwell_strong_transition`'s paper role is
  settled.
- Use the two post hoc reads below to decide whether the branch supports a
  stronger mechanistic claim than “periodic re-encoding prevents stale-chart
  failures.”

## April 1, 2026: Collected Chart-Change Attribution Read

Concrete result(s):
- The post hoc chart-change attribution pass is complete under
  [results/transition_rich_chart_change_attribution_20260401](/home/mila/l/lia/skae/results/transition_rich_chart_change_attribution_20260401).
- On `gated_transfer_linear`, all three model roots show positive periodic gain
  on chart-change windows, but that gain is consistently smaller than the gain
  on non-switch and stable subsets:
  - chart-change-window gain: `0.580229`, `2.5197`, `0.678105`
  - non-switch gain: `3.13667`, `18.4534`, `2.50009`
  - transition-region gain: `0.259326`, `0.591487`, `0.30107`
  - stable-region gain: `3.02077`, `17.7154`, `2.43167`
- All three localization deltas are negative for all three roots:
  - chart-change vs non-switch: `-2.55644`, `-15.9337`, `-1.82199`
  - transition vs stable region: `-2.76144`, `-17.0195`, `-2.14286`
  - transfer-switch vs all transfer trajectories: `-2.19871`, `-20.3617`,
    `-1.79909`

Result in experimental context:
- This stage was run after the Stage 1 screen using each run's selected
  best-periodic mode.
- It was designed to test a stronger causal claim than plain forecasting
  success:
  whether periodic re-encoding helps specifically when the active local chart
  changes on the explicit-transfer toy.

Interpretation:
- The current `gated_transfer_linear` learned-model read does not support that
  stronger localization claim.
- Periodic re-encoding helps on the transfer toy overall, but the gains do not
  concentrate where true chart changes occur.

Project implications:
- Do not write the current transition-rich branch as “re-encoding helps mainly
  at chart switches” based on `gated_transfer_linear`.
- The defensible claim is weaker and cleaner:
  periodic re-encoding matters on the transfer toy, but the benefit is not yet
  localized to true chart-change windows.

Next steps:
- Demote the chart-localization claim in the paper draft unless new evidence is
  collected.
- If that stronger causal claim is still scientifically central, design a
  cleaner benchmark or extend the attribution read to `gated_local_linear`
  before making it a headline statement.

## April 1, 2026: Collected Support Local-Linearity Read

Concrete result(s):
- The post hoc support local-linearity pass is complete under
  [results/transition_rich_support_local_linearity_20260401](/home/mila/l/lia/skae/results/transition_rich_support_local_linearity_20260401).
- `gated_local_linear` is the strongest partition-reuse positive:
  - retained trajectory coverage: `0.83391`, `0.923875`, `0.806228`
  - chart-switch coverage: `0.855491`, `0.907514`, `0.791908`
  - local/global/shuffled `H`-step NRMSE:
    `0.00667185 / 0.0196433 / 0.0534285`,
    `0.00294161 / 0.0127981 / 0.03636`,
    `0.00862393 / 0.0358176 / 0.125676`
- `gated_transfer_linear` is a clear but weaker positive:
  - retained trajectory coverage: `0.581315`, `0.653979`, `0.716263`
  - transfer coverage: `0.4`, `0.7`, `0.6`
  - local/global/shuffled `H`-step NRMSE:
    `0.0453574 / 0.0864248 / 0.15116`,
    `0.0411338 / 0.0843096 / 0.197936`,
    `0.0474659 / 0.0930806 / 0.172619`
  - weighted source-endpoint-pair purity stays high:
    `0.884146`, `0.883495`, `0.888889`
- `multiwell_strong_transition` is positive but weak on coverage:
  - retained trajectory coverage: `0.352941`, `0.418685`, `0.33564`
  - local/global/shuffled `H`-step NRMSE:
    `0.0188171 / 0.0395969 / 0.128213`,
    `0.017011 / 0.0314335 / 0.170924`,
    `0.0173628 / 0.037489 / 0.135196`

Result in experimental context:
- This stage asks whether recurring support groups define reusable local
  partitions on real trained checkpoints, not only whether a root forecasts
  well.
- The crucial controls are the global local-dynamics fit and the shuffled
  support assignment baseline.

Interpretation:
- The answer is yes across the suite:
  recurring support groups recover nontrivial local predictive structure and
  beat both global fits and shuffled assignments.
- The cleanest case is `gated_local_linear`, the transfer toy is intermediate,
  and `multiwell_strong_transition` is weakest.
- The matched zero-sparsity MLP is also strong on this diagnostic, often as
  strong as or stronger than the sparse root.

Project implications:
- The current mechanism claim should be about reusable local partitions, not
  about an effect that only appears with an explicit sparsity penalty.
- This is good paper evidence for interpretability and local predictive
  structure, but it does not currently support a sparse-only mechanism story.

Next steps:
- Write the mechanism result as a three-way comparison, not as a sparse-vs-dense
  binary.
- If a sparse-specific claim is still desired, it will need new evidence beyond
  the current support local-linearity summaries.

## Core Evidence Snapshot

### Transition-Rich Basin Partitioning

1. Concrete results:
- The new tests required by the transition-rich plan now exist and pass:
  - `tests/test_transition_rich_system_determinism.py`
  - `tests/test_transition_rich_endpoint_labels.py`
  - `tests/test_transition_rich_crossing_metrics.py`
  - `tests/test_transition_rich_env_registry.py`
  - `tests/test_transition_rich_calibration_schema.py`
- The local verification commands completed cleanly:
  - `env -u VIRTUAL_ENV uv run python -m pytest tests/test_transition_rich_system_determinism.py tests/test_transition_rich_endpoint_labels.py tests/test_transition_rich_crossing_metrics.py tests/test_transition_rich_env_registry.py tests/test_transition_rich_calibration_schema.py -q` -> `17 passed`
  - `env -u VIRTUAL_ENV uv run python -m pytest tests/test_env_interface.py tests/test_data.py -k 'make_env or multiwell' -q` -> `5 passed`
- The paper-facing trajectory figure now exists at [transition_rich_100_trajectories.svg](/home/mila/l/lia/skae/docs/figures/transition_rich_20260331/transition_rich_100_trajectories.svg), with companion `pdf/png/json` artifacts in [docs/figures/transition_rich_20260331](/home/mila/l/lia/skae/docs/figures/transition_rich_20260331).
- Short training-entry smokes also pass through the real training CLI:
  - `generic_sparse` on `multiwell_strong_transition`
  - `generic_sparse` on `gated_local_linear`
  - matched zero-sparsity MLP smoke on `gated_local_linear`
  - small LISTA smoke on `multiwell_strong_transition`
- Those runs write checkpoints and training plots under [runs/transition_rich_smoke](/home/mila/l/lia/skae/runs/transition_rich_smoke).
- The deterministic calibration tool now exists at [tools/calibrate_transition_system.py](/home/mila/l/lia/skae/tools/calibrate_transition_system.py).
- The two native-plot `2D` paper candidates are now frozen from deterministic calibration on a fixed `17x17` initial-condition grid with a `60`-step finite rollout, `100`-step long-rollout endpoint check, `10`-step extension, and `10`-step settling window:
  - `multiwell_strong_transition`
    - endpoint basins: `5`
    - endpoint distribution: `62 / 62 / 62 / 62 / 41`
    - per-endpoint crossing fractions: `0.500 / 0.500 / 0.500 / 0.500 / 0.585`
    - overall crossing fraction: `0.512`
    - label stability rate: `1.000`
  - `gated_local_linear`
    - endpoint basins: `3`
    - endpoint distribution: `103 / 97 / 89`
    - per-endpoint crossing fractions: `0.670 / 0.577 / 0.539`
    - overall crossing fraction: `0.599`
    - label stability rate: `1.000`
- The explicit-transfer toy is now implemented and calibrated on the fixed
  `17x17` grid with a `180`-step endpoint rollout and `15 + 15` settling /
  extension check:
  - `gated_transfer_linear`
    - endpoint basins: `3`
    - source-neighborhood counts: `9 / 9 / 9`
    - per-source-neighborhood transfer fractions: `0.333 / 0.444 / 0.333`
    - overall source-neighborhood transfer fraction: `0.370`
    - core counts: `3 / 2 / 2`
    - per-core retention fractions: `1.000 / 1.000 / 1.000`
    - overall core retention fraction: `1.000`
    - label stability rate: `1.000`
- Paper-facing transfer-system figure artifacts now exist:
  - [gated_transfer_linear_region_map.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_region_map.svg)
  - [gated_transfer_linear_chart_trajectories.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_chart_trajectories.svg)
  - [gated_transfer_linear_endpoint_trajectories.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_endpoint_trajectories.svg)
  - [gated_transfer_linear_transfer_summary.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_transfer_summary.svg)
- A trajectory-figure audit is now also documented in
  [gated_transfer_linear_trajectory_plot_audit_20260401.md](/home/mila/l/lia/skae/docs/planning/gated_transfer_linear_trajectory_plot_audit_20260401.md).
  It records that the original figure packet used a uniform `10x10` grid, that
  only `8/100` plotted starts lie in source neighborhoods, and that clearer
  ground-truth trajectory views were added:
  - [gated_transfer_linear_uniform_start_grid_10x10.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_uniform_start_grid_10x10.svg)
  - [gated_transfer_linear_ground_truth_trajectories_10x10_starts.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_ground_truth_trajectories_10x10_starts.svg)
  - [gated_transfer_linear_ground_truth_source_starts_17x17.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_ground_truth_source_starts_17x17.svg)
  - [gated_transfer_linear_dense_source_trajectories.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_dense_source_trajectories.svg)
  - [gated_transfer_linear_flow_field.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_flow_field.svg)
  The dense-source panel uses `144` starts inside the three source
  neighborhoods, and the flow-field panel makes clear that the line-like
  transport geometry is part of the ground-truth system itself rather than a
  sampling artifact.
- The `multiwell_gradient` reference under the same corridor-aware labeling is not yet the clean low-transition control:
  - endpoint basins: `5`
  - per-endpoint crossing fractions: `0.433 / 0.422 / 0.422 / 0.410 / 0.242`
  - overall crossing fraction: `0.401`
- Undertrained support-local-linearity smoke validation now also exists on the
  transfer toy under
  [results/transition_rich_smoke_support_local_linearity_20260401](/home/mila/l/lia/skae/results/transition_rich_smoke_support_local_linearity_20260401):
  - on the collected `generic_sparse` smoke, retained support groups cover
    `0.858` of all trajectories, `0.865` of chart-switch trajectories, and
    `0.700` of transfer trajectories
  - `H20` latent NRMSE is `0.0701` for local fits versus `0.1309` for the
    global baseline and `0.2018` for shuffled assignments
  - weighted endpoint purity is `1.000` and weighted source-endpoint-pair
    purity is `0.891`

2. Context:
- The benchmark, hard-system, and mechanism packets are now strong enough to act as supporting evidence while this new branch is screened.
- The calibration pass used only ground-truth environment dynamics and ground-truth region labels:
  - `multiwell_*` now exposes explicit basin cores plus a central transition corridor
  - `gated_local_linear` exposes explicit basin cores and gate sectors with exact local Jacobians
- No forecasting model was used to define or validate these labels.
- The 100-trajectory figure verifies that both frozen systems are native-plot `2D` systems with visually interpretable mechanics, not only numerically acceptable calibration summaries.
- The support-local-linearity numbers above are still tooling-validation reads
  on `5`-step smoke checkpoints, not paper-facing scientific claims; they are
  useful because they verify that the new post hoc stage is nondegenerate on
  real artifacts and that it can see transfer/chart-switch trajectories rather
  than collapsing to easy stable-only subsets.
- Updated interpretation:
  - `gated_local_linear` is the cleanest oracle-validated chart-switching toy
  - `multiwell_strong_transition` is the weaker shared-corridor toy
  - `gated_transfer_linear` is the explicit-transfer stress test rather than
    the sole mechanistic flagship

3. Interpretation:
- System design is now closed at the toy-environment level for the lead branch.
- The branch now has one clean mechanistic positive, one harder explicit-
  transfer stress test, and one weaker shared-corridor toy, all with stable
  endpoint labels and explicit region charts.
- The remaining open question is whether the learned supports on these systems yield reusable, label-light partitions that beat simple controls.
- `multiwell_gradient` should stay secondary until we decide whether we need to retune it into a genuinely low-transition control.
- The real training CLI accepts the new systems cleanly, so the screening trio can proceed without more environment/plumbing work.
- The training and analysis tooling is now strong enough to run a real
  seed-robust screening pass rather than another provisional pilot.
- The branch also now has a clean post hoc mechanistic tool for testing whether
  periodic gains localize near true chart changes, without contaminating the
  live Stage 1 training queue with mixed evaluation code.
- The branch now also has a clean post hoc support-local-linearity tool for
  testing whether recurring support groups define reusable local charts on the
  same deterministic grid.

4. Project implications:
- Environment design is no longer the rate-limiting step.
- The first model pass is now concretely defined as the full `27`-run Stage 1
  screen at the paper budget, with the oracle benchmark now fixing the expected
  role split:
  - `gated_local_linear` as the clean mechanistic positive
  - `gated_transfer_linear` as the transfer stress test
  - `multiwell_strong_transition` as the weaker secondary toy
- The clean next interpretive read is now also fixed:
  use the post hoc chart-change attribution diagnostic on finished
  `gated_local_linear` and `gated_transfer_linear` runs to test whether
  periodic gains are actually localized at true chart switches.
- A second post hoc interpretive read is also fixed:
  use the transition-rich support-local-linearity stage on finished runs to
  test whether recurring support groups cover transfer/chart-switch
  trajectories and beat global/shuffled local-dynamics baselines.
- Primary diagnostics should emphasize partition reuse, local predictive structure, and transition handling; basin-label metrics remain benchmark-only secondary reads.
- Forecast-side diagnostics should now explicitly include deterministic flow consistency: nearly identical full states should not branch to materially different futures except at the tolerance level already observed in the ground-truth simulator.
- Forecast sanity is now a post-freeze screening metric rather than a toy-system design gate.

5. Next steps:
- Use the completed collected summaries above as the active evidence for this
  branch rather than treating the screen as pending.
- Center the paper-facing transition-rich story on:
  - `gated_local_linear` as the clean mechanistic positive
  - `gated_transfer_linear` as the transfer stress test
  - `multiwell_strong_transition` as a weaker secondary toy unless a targeted
    follow-up rescues it
- Do not make the stronger chart-localization claim on the current
  `gated_transfer_linear` evidence.
- Write the support-local-linearity result as a reusable-partitions positive
  that is not sparse-specific under the current controls.
- Decide whether to rerun the single missing dense-LISTA `multiwell` seed only
  after the final paper role of `multiwell_strong_transition` is settled.

### Cross-System Forecasting

1. Concrete results:
- The live benchmark headline is the verified `29`-system, fair-`200k` packet under `results/paper_zero_sparse_benchmark_200k_20260321`, together with the aligned canonical Kuramoto packet under `results/kuramoto_dt0p01_200k_canonical_20260323`.
- Best-periodic cross-system medians at `H100/H500/H1000/H1500/H2000/H2500/H3000` are:
  - sparse MLP: `2.947e-4 / 0.0051 / 0.0240 / 0.0795 / 0.1123 / 0.1591 / 0.2201`
  - zero-sparsity MLP: `3.021e-4 / 0.0063 / 0.0353 / 0.1067 / 0.1351 / 0.1738 / 0.1951`
  - dense LISTA: `3.440e-4 / 0.0047 / 0.0250 / 0.0449 / 0.0627 / 0.0880 / 0.1039`
- Good-system counts are:
  - sparse MLP: `27/25/25/26/25/25/25`
  - zero-sparsity MLP: `27/26/26/27/27/27/27`
  - dense LISTA: `28/25/26/25/25/25/25`
- Under one fixed `periodic_100` cadence, late-horizon medians at `H1500/H2000/H2500/H3000` are:
  - sparse MLP: `0.1070 / 0.2140 / 0.3690 / 0.6148`
  - zero-sparsity MLP: `0.1729 / 0.4542 / 0.7041 / 0.9349`
  - dense LISTA: `0.0744 / 0.2688 / 0.4437 / 0.6508`

2. Context:
- This is the benchmark family retained in `docs/review_main_results_tables_20260314.tex`.
- The dense Stage 1-4 tuning chain is appendix-only provenance for fixing the dense comparator and no longer belongs in the live paper-facing evidence.

3. Interpretation:
- Sparse MLP is still the clean anchor at `H100` and `H1000`.
- Dense LISTA is median-best at `H500` and from `H1500` through `H3000`.
- The zero-sparsity MLP broadens coverage on some later horizons, but it is not the benchmark-median winner at any reported horizon.
- Under one fixed cadence, explicit sparsity is still materially better than the matched zero-sparsity MLP, but dense LISTA remains the late-horizon median winner.

4. Project implications:
- The benchmark claim must now be written as a genuine three-way result.
- Do not reduce the benchmark story to “sparse beats dense” or “dense beats sparse.”
- Use this packet as the supporting benchmark anchor while the transition-rich branch is built.

5. Next steps:
- Keep the fair `200k` packet as the supporting benchmark anchor.
- Keep the dense recipe-selection chain and the old `50k` audit in `docs/EXPERIMENTS_ARCHIVE.md`.

### Hard-System Forecasting

1. Concrete results:
- Kuramoto remains the clearest targeted structured result. At `dt=0.00625`, `lista_blockdiag` rescues the `N=16` and `N=32` settings relative to the sparse anchor, but the benefit is selective and does not survive uniformly across the full family.
- The repaired block-diagonal MLP fairness controls are now the valid source of truth:
  - uniform-spread `N=16`: `generic_sparse_blockdiag = 8.13 / 24.76 / 91.23 / 399.74 / 1724.45` at `H1000/H1500/H2000/H2500/H3000`
  - dimension sweep `H1000`, `N=8/16/24/32/64`: `10.61 / 6.51 / 5.79 / 5.16 / 208.54`
- The repaired `N=64` block-diagonal MLP row is a real seed-fragile limit, not a provenance bug: `2/5` good seeds and a seed-median `H1000 = 208.54`.
- Hopfield stays negative for LISTA-family claims. Smaller `dt` improves errors, but sparse MLP remains best through `H2000` at `N=16`, and the quarter-step `N=64` probe remains strongly negative.
- Corrected competitive Lotka-Volterra is no longer a forecasting holdout. The corrected `4`-basin row is negative for a clean block-structured win, and the fixed `8`-basin follow-ups split by `dt` rather than yielding one stable architecture reversal.

2. Context:
- This section merges the smaller-`dt` rescues, Kuramoto robustness checks, repaired fairness controls, Hopfield quarter-step probe, and corrected competitive-LV follow-ups into one hard-system family.
- The historical MLP `+ block_diagonal K` rows are invalid and must not be reused.

3. Interpretation:
- Smaller `dt` is the dominant hard-system lever, but it does not create a uniform LISTA success story.
- Block structure helps selectively on Kuramoto, fails at `N=64`, and does not rescue Hopfield.
- Removing the sparsity penalty can improve some hard-system rows, so the hard-system narrative must stay selective rather than global.

4. Project implications:
- Keep `lista_blockdiag` scoped to targeted Kuramoto claims.
- Present Hopfield as an MLP-better limitation case and corrected competitive-LV as a mixed forecasting result with weak mechanistic follow-through.
- Use this family as limitation/support evidence rather than as a live execution branch.

5. Next steps:
- Do not queue new broad hard-system sweeps by default.
- If a figure is needed, prioritize one that shows the Kuramoto horizon split and the `N=64` failure boundary explicitly.

### Basin-Support And Mechanism

1. Concrete results:
- The broad support audit and label-free clustering family are now stable enough to write as one system-dependent mechanism section:
  - multiwell: strong positive
  - Duffing: weak
  - Kuramoto: negative
  - Hopfield: continuous separation without reusable sparse supports
  - corrected competitive-LV: negative
- The direct Kuramoto support audit remains the key hard negative. Every family still shows essentially trajectory-unique supports, low basin consistency, flat Hamming geometry, and no reusable non-singleton mode supports.
- The matched zero-sparsity MLP does not fix Kuramoto support reuse.
- The recurring-support local-linearity study is partial on `multiwell_strong_transition` and negative on targeted smaller-`dt` Kuramoto:
  - multiwell retained coverage stays below the planned acceptance gate (`0.4141-0.4844` depending on root)
  - Kuramoto retains `0` recurring support groups across the evaluated root-seed cases
- Corrected competitive-LV stays negative-to-weak: no clean paper root clears the support-view gate, cosine separation remains negative, and clustering collapses to unstable discovered group counts rather than stable basin recovery.

2. Context:
- This family is retained because it directly supports the basin-support alignment goal and the mechanism section in `docs/review_main_results_tables_20260314.tex`.
- Binary support uniqueness is no longer a paper headline metric because it saturates even on negative systems.

3. Interpretation:
- Explicit sparsity acts as a representation-shaping prior on some systems, especially multiwell, but it does not yield a universal basin-support win.
- Kuramoto remains the decisive negative result for support-defined regime reuse, and that negative survives the matched zero-sparsity control.
- Corrected competitive-LV narrows rather than broadens the mechanism claim.

4. Project implications:
- Write the mechanism section as system-dependent and bounded.
- Keep cosine separation, label-free recovery, and recurring-support local-linearity ahead of binary uniqueness in the paper narrative.
- Use this family as the metric starting point for the new transition-rich branch rather than as the branch endpoint.

5. Next steps:
- If one more mechanism analysis is needed, make it a small offline sensitivity or a targeted seed extension on the undercovered corrected competitive-LV support-alignment branch.
- Do not reopen broad representation sweeps by default.

## Recent Paper-Critical Updates

### March 31 Tests And Calibration Freeze Two Transition-Rich 2D Candidates

1. Concrete results:
- The transition-rich implementation pass is complete locally:
  - the required tests exist and pass
  - the new calibration module is in [skae/transition_calibration.py](/home/mila/l/lia/skae/skae/transition_calibration.py)
  - the reproducible calibration entry point is in [tools/calibrate_transition_system.py](/home/mila/l/lia/skae/tools/calibrate_transition_system.py)
- The paper-facing 100-trajectory figure now exists at [transition_rich_100_trajectories.svg](/home/mila/l/lia/skae/docs/figures/transition_rich_20260331/transition_rich_100_trajectories.svg).
- Short training-entry smokes also completed cleanly under [runs/transition_rich_smoke](/home/mila/l/lia/skae/runs/transition_rich_smoke).
- Two native-plot `2D` systems are now frozen as paper candidates from deterministic calibration on the fixed `17x17` grid:
  - `multiwell_strong_transition`: `5` endpoint basins, crossing fractions `0.500 / 0.500 / 0.500 / 0.500 / 0.585`, overall crossing `0.512`, label stability `1.000`
  - `gated_local_linear`: `3` endpoint basins, crossing fractions `0.670 / 0.577 / 0.539`, overall crossing `0.599`, label stability `1.000`
- The same calibration shows that `multiwell_gradient` is not yet the desired low-transition control under the new corridor-aware labeling (`overall crossing = 0.401`, center-basin crossing `0.242`).

2. Context:
- This is the first time the branch has concrete deterministic systems that satisfy the basin-count and transition-richness gates before any model training.
- The endpoint labels were checked only with ground-truth rollouts (`100` steps plus a `10`-step extension), not with model predictions.

3. Interpretation:
- The toy-system design problem is now solved for the positive pair.
- The lead blocker has shifted to model screening and diagnostic readout rather than further environment implementation.

4. Project implications:
- The branch can move directly to the screening trio and the label-light partitioning diagnostics.
- Forecasting becomes a supporting post-freeze read rather than the criterion used to design the systems.
- No additional environment integration work is needed before those screening runs.

5. Next steps:
- Train the screening trio on the two frozen systems.
- Add the sparse-anchor forecast sanity metric back into the calibration summary once the first sparse-anchor checkpoints exist.

### March 31 Paper Direction Shifts To Transition-Rich Basin Partitioning

1. Concrete results:
- The live paper direction is now frozen in [docs/planning/transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md).
- The fair `200k` benchmark packet, the hard-system packet, and the current mechanism packet are now treated as supporting evidence rather than the lead live branch.
- The new lead branch is deterministic transition-rich basin partitioning and classification on analyzable toy systems.
- The first-pass constraints are fixed: tests first, native-plot `2D` first, `3-10` endpoint basins, and acceptable per-endpoint-basin crossing fractions in the `0.30-0.70` range.

2. Context:
- This shift follows closure of the broad forecasting queue and stabilization of the current mechanism read.
- The remaining paper leverage is no longer another benchmark rerun; it is a cleaner narrative driver with controlled deterministic dynamics.

3. Interpretation:
- The project is no longer blocked on collecting more broad forecasting rows.
- It is blocked on building a toy-system branch that lets us study partition reuse, transitions, and forecasting failure modes mechanistically.

4. Project implications:
- Immediate work should move to tests, minimal interface scaffolding, candidate calibration, and the first screening trio.
- MSE becomes supportive rather than sufficient; the main new metrics should diagnose why a model works or fails.

5. Next steps:
- Implement the plan in order: tests, scaffolding, system dynamics, calibration, candidate freeze, and screening.
- Keep the existing benchmark, hard-system, and mechanism packets fixed while this new branch is built.

### March 31 Raw-Source Seed-Statistics Reanalysis Adds Robust Seed Summaries

1. Concrete results:
- A new raw-source report now exists at `docs/PAPER_SEED_STATISTICS_20260331.md`, with machine-readable tables under `results/paper_seed_statistics_20260331`.
- The reanalysis recomputes paper-facing results from the lowest-level surviving per-seed artifacts: `evaluation_results_best.json`, `support_alignment.json`, `analysis_results.json`, and local-linearity `metrics.json`.
- Raw-vs-collector verification passed across every family where both sources exist, including `27,060` checked label-free clustering cells and `12,180` checked benchmark best-periodic forecasting cells.
- The raw finite-value coverage audit is stricter than the earlier collector-row audit: benchmark best-periodic stays clean, but fixed-cadence late-horizon Hopfield and embedded-multiwell rows drop below `10` finite seeds, corrected `4`-basin CLV block-diagonal LISTA drops to `14` finite seeds at `H1000-H3000`, repaired Hopfield `N=64` block-diagonal MLP drops to `7` finite seeds at `H1500-H3000`, the direct Kuramoto mode-support audit stays at `5` seeds, and corrected competitive-LV support alignment stays at `3`.

2. Context:
- This pass was triggered to replace middle-seed medians with mean, standard deviation, IQM, and bootstrap confidence intervals computed directly from the original per-seed JSON outputs wherever those still exist.

3. Interpretation:
- The robust summaries are now provenance-clean.
- They also show that “collector row exists” and “finite seed statistic exists” are not always the same statement on the late-horizon tails.

4. Project implications:
- Use `docs/PAPER_SEED_STATISTICS_20260331.md` for any discussion that needs average-case rather than median-only reporting.
- Do not describe the entire paper-facing packet as uniformly `10`-seed clean without the raw finite-value caveat.

5. Next steps:
- Keep the new report as the seed-statistics companion to the senior-coauthor packet.
- If any coverage repair is queued, prioritize only the remaining mechanism gaps or the clearly identified late-horizon finite-value holes.

### March 30 Verification Pass Closes Forecasting Seed-Coverage Gaps

1. Concrete results:
- The table-facing forecasting packet used by `docs/review_main_results_tables_20260314.tex` is now seed-clean: every displayed benchmark and hard-system forecasting row uses at least seeds `0-9`, and corrected `4`-basin competitive Lotka-Volterra uses seeds `0-14`.
- The residual Kuramoto repair chain `9074821 -> 9074822 -> 9074823` filled the last missing forecasting seed (`generic_sparse_sc0_n8`, seed `6`).
- The March 31 raw-source reanalysis refined this collector-row picture: the remaining mechanism undercoverage is corrected competitive-LV support alignment (`0,1,2`) plus the direct Kuramoto mode-support audit (`5` seeds), and several late-horizon forecasting slices still fall below full finite-value coverage.

2. Context:
- This pass was a direct verification against the exact rows cited in the senior-coauthor handoff tables.

3. Interpretation:
- Forecasting coverage is no longer an execution blocker.
- The remaining undercoverage sits in a mechanism artifact, not in the paper-facing forecasting packet.

4. Project implications:
- The paper can now cite the forecasting packet as seed-clean.
- The remaining blocker is narrative accuracy, not missing forecasting seeds.

5. Next steps:
- Keep any future coverage work narrowly targeted to the undercovered corrected competitive-LV mechanism branch.

### March 21-23 Matched Zero-Sparsity MLP Campaign Closes Across Every Paper Family

1. Concrete results:
- The matched zero-sparsity MLP baseline is fully collected across benchmark, hard-system, and mechanism families under:
  - `results/paper_zero_sparse_benchmark_200k_20260321`
  - `results/zero_sparse_hard_systems_20260321`
  - `results/zero_sparse_mechanisms_20260321`
- The repaired zero-sparsity Kuramoto support-audit rerun confirms that removing explicit sparsity does not rescue reusable Kuramoto basin supports.

2. Context:
- This is the direct causal control for whether the sparsity penalty itself matters, because it keeps the same MLP encoder/decoder family and optimizer recipe as the sparse anchor while setting `lambda_sparse = 0`.

3. Interpretation:
- The zero-sparsity control materially changed the paper position.
- The paper can no longer present a simple sparse-versus-dense story, and it also cannot present zero sparsity as a new global winner.

4. Project implications:
- Keep the zero-sparsity MLP in the benchmark, hard-system, and mechanism tables as a first-class paper control.

5. Next steps:
- Use the three-way sparse-vs-zero-sparse-vs-LISTA framing consistently across all paper-facing docs.

### March 17-20 Repair Wave Replaces The Invalid Historical Block-`K` MLP Controls

1. Concrete results:
- A March 17 audit showed that the historical MLP `+ block_diagonal K` controls were invalid because `GenericKM` ignored `K_STRUCTURE` and always learned a dense latent transition.
- The repair wave completed cleanly and now provides the valid block-`K` MLP controls used in the hard-system family and the senior-coauthor packet.
- The March 19 retry wave closed the remaining Kuramoto mirror gaps, and the March 20 audit confirmed that the bad `N=64` repaired row is a real seed-fragility pattern rather than a stale artifact.

2. Context:
- These controls matter because they isolate whether a reported gain comes from encoder structure or only from imposing block structure on the latent transition.

3. Interpretation:
- Historical MLP `+ block-K` rows remain invalid provenance and should stay retired.
- The repaired controls show a selective picture: real positives on some Kuramoto settings, negative or mixed results elsewhere.

4. Project implications:
- All causal language about block structure must use the repaired March 17-20 outputs rather than the earlier historical rows.

5. Next steps:
- Keep the repaired controls as the only paper-facing source for block-`K` fairness claims.

## Archive Pointer

- `docs/EXPERIMENTS_ARCHIVE.md` now holds the appendix-only dense tuning chain, the old `50k` benchmark audit, superseded queue snapshots, intermediate CLV and Kuramoto follow-up chronology, and other lower-priority paper-era detail that no longer belongs in the live file.
