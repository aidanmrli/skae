# Paper Track Status

Date: April 28, 2026
Evidence organization last refreshed: `2026-04-28 12:06 EDT`
Paper-critical live queue status last refreshed: `2026-04-28 19:09 EDT`

## Paper-Track Summary

Problem being solved:
- The paper now needs an evidence-first experiments narrative matching the
  draft setup: multibasin Koopman learning, sparse supports as inspectable
  support objects, model-produced support objects, and support-based routing.
- The narrative must make a sharp distinction between two claims: a support can
  identify where a state is, while a stronger and separate test is whether that
  support selects useful latent coordinates or a useful local linear law for
  prediction.

Current solution:
- Use [PAPER_EXPERIMENT_EVIDENCE_MAP.md](/home/mila/l/lia/skae/docs/PAPER_EXPERIMENT_EVIDENCE_MAP.md)
  as the drafting order and display plan for the experiments section.
- Keep the main paper evidence in this order:
  1. support agreement with basin labels, with labels used only for evaluation;
  2. non-oracle support-routed local prediction using the same support objects;
  3. long-horizon forecasting competitiveness;
  4. supporting and falsification diagnostics.
- Build the main displays in the same order: support maps, fixed-`17`
  alignment/forecasting table, non-oracle routing table, support
  refresh/routing figure, and Dysts long-horizon table.
- Keep active supports, not pre-specified latent blocks, as the primary
  interpretability object. Do not reframe the paper around basin-block
  alignment or around training-time access to basin counts.
- State explicitly that support-label agreement is necessary but insufficient:
  a support may be an excellent basin label while the predictive dynamics are
  carried by different latent coordinates, continuous coefficient values, or
  cross-coordinate couplings in the learned Koopman transition.
- Tables 2-4 now use within-system confirmatory tests rather than cross-system
  significance tests: seed-paired Wilcoxon/Holm for non-oracle routing and
  Dysts forecasting, and transfer-pair Wilcoxon/Holm for support refresh. The
  completed jobs `9388212-9388218` are per-basin deep-slice interpretability
  outputs for Table 1 appendix robustness; they are not Table 2-4 seed
  packets.

Outstanding problem:
- The remaining paper work is presentation and claim calibration, not broad
  evidence discovery: build the main basin/support figure, the routing table,
  the support-refresh/routing figure, and the Dysts long-horizon table; keep
  true local-geometry recovery as a mixed secondary diagnostic; decide whether
  the narrowed dense exact top-`8` support-refresh claim needs seed or
  threshold robustness before submission.
- The per-basin deep-slice interpretability rerun is complete, not queued:
  jobs `9388212-9388218` all completed with exit `0:0`. Use it as an appendix
  robustness check for Table 1 coverage, while keeping the global deep-slice
  Table 1 numbers as the main-text source of truth.
- The Table 2-4 statistical procedure is no longer ambiguous. Table 2 now
  displays the `H100` routed/global ratio with within-system seed-paired
  Wilcoxon/Holm `[K/17]` counts because the current seed-`0`--`9` packet gives
  strong confirmatory counts there (e.g. LISTA-SB exact-support routes clear
  `12/17`, `11/17`, `14/17`, and `15/17` across all/deep gated/local cells).
  The `H1000` version remains underpowered because route availability leaves
  only `1`--`10` valid seed pairs in many systems; the Table 2 caption has a
  `\todo{}` to revisit `H1000` after the seed expansion. Table 3 support
  refresh uses transfer pairs within system and remains strong (`11/12`,
  `11/12`, `10/12`, `10/12` across the four displayed cells). Table 4 stays
  seed-paired Wilcoxon/Holm against Dense MLP and remains underpowered at
  `n_seed=10` until the live `n_seed=15` Dysts rerun completes.
- The apparent Table 2 seed-`10`--`14` job array `9392598` completed, but it
  is not the requested paper-facing expansion: it covers only three roots and
  `20k` steps. The outstanding Table 2 expansion is still five models
  (`LISTA-SB`, `LISTA-BD`, Dense MLP, Sparse MLP, Sparse MLP-BD), `200k`
  steps, seeds `10`--`14`, followed by self-routed forecasting/statistical
  regeneration.
- Status update at `2026-04-28 16:27 EDT`: the Table 2 five-model path is now
  being repaired in two stages. Existing `200k` hard-init MLP-control seeds
  `0`--`9` are queued for self-routed evaluation as shard jobs
  `9395314`--`9395319` with merge job `9395320`; dependent stats-refresh job
  `9395334` will rerun the per-system paired tests after that merge. The full
  training backfill is staged as queue job `9395321`; it will submit a
  `433`-task `%64` GPU array once the expanded user job count drops below
  `550`. The task set is seeds `10`--`14` for all five Table 2 roots plus the
  `8` known missing seed-`0`--`9` hard-init MLP-control rows.
- Status update at `2026-04-28 16:40 EDT`: the main fairness caveat on the
  current LISTA-SB row is now explicitly queued as a sensitivity. Queue job
  `9395415` runs
  `scripts/queue_transition_rich_lista_sb_p256_hardinit_fairness.sh`, which
  will create and submit a `255`-task `%64` GPU array for
  `lista_dense_softblock_signsplit_p256_hardinit_basin_partition`: the same
  hard-init dense soft-block sign-split recipe as LISTA-SB, but with
  `target_size=256` instead of `64`, across `17` systems and `15` seeds. The
  launcher also queues forecasting collection and `topk:8` self-routed
  forecasting after training.
- 2026-04-28 Dysts seed-15 / `seq_len=10` / 12-system re-train is in flight
  via chunked jobs: chunk 1 is `9392814`, and chunk 2 was submitted by
  replacement orchestrator `9393138` as `9393590` at 2026-04-28 12:51 EDT.
  As of 2026-04-28 15:03 EDT, expanded child-task state still shows both
  arrays running/pending under their `%64` throttles, so use `squeue -r`
  child tasks rather than the top-level array state for completion checks.
  Training samples windows of length `10` from 30K-step Dysts source
  trajectories. After both chunks land, the Dysts
  long-horizon eval will run on a separate `long60` held-out test cache
  (`steps=60000`) with horizons
  `H5000/H10000/H20000/H30000/H40000/H50000/H60000` and re-encode periods
  `{50, 75, 100, 200, 400, 600, 1000}`. Table 4 will be redrawn against this
  new, narrower system list (drops `Duffing`, `SprottTorus`,
  `RikitakeDynamo`) and interpreted as forecasting beyond the 30K training
  source horizon.
- A Dysts LISTA soft-block `d_z=256` sensitivity row has also been queued using
  [scripts/queue_dysts_seq10_lista_softblock_p256_seeds0to14.sh](/home/mila/l/lia/skae/scripts/queue_dysts_seq10_lista_softblock_p256_seeds0to14.sh).
  Launcher job `9396890` completed with exit `0:0`, built the `180`-task table,
  and submitted training array `9396894` plus dependent long-horizon eval
  launcher `9396895`. It matches the new Table 4 scope (`12` Dysts systems,
  seeds `0`--`14`, `200k` steps, `sequence_length=10`, sparsity coefficient
  `0.006`) and uses a dense LISTA sign-split encoder with a 16-block soft-block
  transition penalty at `d_z=256`. Treat it as an add-on sensitivity until its
  training and `long60` evaluation chain complete.
- The paper-facing prose should always explain why the next
  experiment follows: label agreement asks whether the support says where the
  state is; routing asks whether the support helps predict where it goes next.

## Goal

Fixed-`17` LISTA root/result lookup:
- Use [FIXED17_LISTA_RESULTS_INDEX.md](/home/mila/l/lia/skae/docs/FIXED17_LISTA_RESULTS_INDEX.md) as the canonical quick-reference page for the fixed-`17` LISTA roots, their packets, and the headline results that currently matter for the paper write-up.
- Use [SUPPORT_OBJECT_GLOSSARY.md](/home/mila/l/lia/skae/docs/SUPPORT_OBJECT_GLOSSARY.md) for the paper-facing definitions of `absolute:0.001`, `relative:0.1`, `topk:8`, exact support, support family, and dominant group.

The paper target is now explicit:

- make the lead live branch a fixed-`17` basin-separation comparison:
  do the models learn reusable sparse supports for distinct basins on the
  selected systems, and is some induced sparsity essential for good
  finite-dimensional Koopman representations when multiple basins or fixed
  points coexist? LISTA is one structured way to induce that sparsity, not
  the claim by itself.
- keep the fair `200k` forecasting packet, the hard-system packet, and the
  existing mechanism packet as **decision-grade supporting evidence** rather
  than the lead live branch
- execute that branch on the 17 systems that we can analyze mechanistically in the paper through the plan in [docs/planning/transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md)
- use [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md) as the current ground-truth planning note for ablation design choices when iterating on items `3` and `4` of that transition-rich plan, and replace planning assumptions with experiment-backed conclusions once those axes are run systematically
- distinguish **endpoint basin** from **finite-horizon transition**; the new branch is intentionally transition-rich even though endpoint basins remain well defined
- write **tests before any system-specific code**, then calibrate toy systems before queueing model sweeps
- keep training-time method design label-free: do not assume known basin counts or basin labels outside benchmark diagnostics
- prioritize support agreement with basin labels, support-view clustering,
  recurring-support reuse, and local predictive structure over MSE-only
  reporting
- measure long-horizon forecasting at `H100`, `H500`, and `H1000` as a
  downstream functional test of the support-label hypothesis: if sparsity
  helps because the Koopman state retains basin identity and that identity is
  dynamically useful, forecasting should be strongest away from separatrices
  and weakest near separatrices where the relevant local law is ambiguous
- use oracle basin-depth / separatrix-proximity slices only for benchmark
  evaluation; do not turn those signals into training-time assumptions
- treat local-law interpretability as basis-aware and symmetry-aware: do not require different basin eigenvalues as the success criterion; instead compare raw and similarity-aligned operator/Jacobian families together with support-family uniqueness up to alignment
- treat the local-versus-global evidence as currently strongest against the
  trained model's learned global transition, not against every possible global
  centered refit. The centered local-law packet supports support-conditioned
  laws on covered states, especially away from basin boundaries, but the
  separately refit global-centered comparison remains weaker and should be
  written cautiously.
- use the two reviewer-response branches as falsification diagnostics:
  [true_jacobian_geometry_experiment_20260423.md](/home/mila/l/lia/skae/docs/planning/true_jacobian_geometry_experiment_20260423.md)
  tests true local-geometry agreement, and
  [controlled_transfer_switching_experiment_20260423.md](/home/mila/l/lia/skae/docs/planning/controlled_transfer_switching_experiment_20260423.md)
  tests support switching under deliberate basin transfer. A tabula-rasa audit
  found evaluator mistakes in the first seed-`0` fixed-`17` outputs, and those
  first outputs are superseded. Corrected reruns are now complete. The
  true-geometry result is a secondary, mixed falsification diagnostic:
  LISTA support families often beat random count-matched partitions near
  attractors, but the zero-sparse MLP often has lower absolute
  projected-Jacobian error because its latent representation is closer to the
  identity map. The
  controlled-transfer result is sharper: dense LISTA exact `topk:8` supports
  switch well after a deliberate state-space bridge, while zero-sparse MLP
  exact supports remain weak; support-family switching is strong for all roots
  and therefore is not sufficient for a LISTA-specific sparsity claim.
- keep the matched standard MLP encoder control as the main causal comparator
  on this branch; use the clean tanh / no-shrink MLP when isolating the
  induced-sparsity effect directly
- for paper-facing comparisons that claim an architecture effect, keep the
  training sampling regime matched across compared roots; treat hard-init
  oversampling as a separate factor and report standard versus hard-init as
  separate table entries rather than folding them into one causal read
- do not treat local-switch localization or sparse-only mechanism claims as the branch acceptance criterion; those are supporting context, while the live branch question is basin separation on the fixed `17` systems
- treat **`200k` as the only main-text training budget** for the frozen benchmark and hard-system supporting packets and this should only be done once we have compelling and significant results at a 20k budget
- use **`20k` as the working training budget** for forward interpretability and metric-diagnosis runs on the fixed `17`-system shortlist while the recipe is still moving, and reserve `200k` on that branch for the final locked confirmatory rerun plus the now-completed narrow default-sampling forecast-floor check on the best `v6` / `v7` roots
- use **`1` seed as the default working budget** for forward interpretability diagnostics on the fixed `17`-system shortlist while the branch is still choosing methods and metrics, and expand to `10` seeds only after a seed-`0` result looks strong enough to justify a paper-facing robustness check
- use the **default sampling regime** for forward LISTA comparator sweeps on
  the fixed `17` systems so the comparison against the MLP controls does not
  inherit a sampling confound; keep hard-init or other altered sampling
  schemes as separate ablation rows once the base recipe is fixed
- treat the dense LISTA Stage 1-4 chain as **appendix-only comparator-selection provenance**
- the MLP `+ block_diagonal K` fairness controls were rerun on March 17/19; full audit: [docs/mlp_block_k_audit_20260317.md](/home/mila/l/lia/skae/docs/mlp_block_k_audit_20260317.md)

Active execution note:
- The forecasting packet is now decision-grade, and the raw-source seed-statistics companion report is in [docs/PAPER_SEED_STATISTICS_20260331.md](/home/mila/l/lia/skae/docs/PAPER_SEED_STATISTICS_20260331.md). It verifies raw-vs-collector agreement and records the remaining raw finite-value coverage gaps explicitly.
- The true-Jacobian/eigendirection and controlled-transfer support-switching
  branches are now scaffolded, revised, smoke-tested, audited, corrected, and
  rerun for fixed-`17` seed-`0`. Implemented files:
  [tools/evaluate_transition_rich_true_jacobian_geometry.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_true_jacobian_geometry.py),
  [tools/evaluate_transition_rich_controlled_transfer_switching.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_controlled_transfer_switching.py),
  [run_transition_rich_true_jacobian_geometry.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_true_jacobian_geometry.sh),
  and
  [run_transition_rich_controlled_transfer_switching.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_controlled_transfer_switching.sh).
  Shell syntax checks pass for the wrappers, and compute-node py-compile
  validation passed after the audit fixes. Corrected smoke jobs `9347587` and
  `9347588` completed with `COMPLETED 0:0`. Corrected true-geometry job
  `9347593` completed in `17m38s` with `49/49` runs, `62,460` rows,
  `30,014` ok rows, and `0` failures under
  [results/true_jacobian_geometry_fixed17_seed0_20260423_corrected](/home/mila/l/lia/skae/results/true_jacobian_geometry_fixed17_seed0_20260423_corrected).
  Corrected controlled-transfer coverage jobs `9347590-9347592` completed
  with `1,776` total rows, `1,632` ok rows, `144` skipped rows, and `0`
  failures under
  [results/controlled_transfer_switching_fixed17_seed0_20260423_corrected](/home/mila/l/lia/skae/results/controlled_transfer_switching_fixed17_seed0_20260423_corrected).
  Interpretation is now claim-calibration rather than execution coverage:
  true geometry is not a headline result, while dense LISTA exact `topk:8`
  controlled-transfer switching is a useful support-switching diagnostic.
  April 25 verification: the second-audited `20260424_reaudit` jobs
  `9347926-9347929` all completed with exit `0:0`, no matching jobs are
  currently queued, and the controlled-transfer packet remains organized as
  three completed root shards rather than a merged top-level summary file.
  Claim-framing note: the MLP roots should be treated as optional specificity
  controls for these two branches, not as required comparators. The geometry
  question can be written as a LISTA-only support/family versus random and
  attractor/basin-baseline diagnostic. The controlled-transfer packet shows
  encoder support switching along a measured state-space basin transfer; a
  stronger periodic-reencoding claim should be phrased as a rollout mechanism
  or backed by an explicit no-reencoding versus reencoding ablation.
- That explicit ablation is now queued. New files:
  [evaluate_transition_rich_periodic_support_refresh.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_periodic_support_refresh.py),
  [run_transition_rich_periodic_support_refresh.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_periodic_support_refresh.sh),
  [queue_transition_rich_periodic_support_refresh.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_periodic_support_refresh.sh),
  and
  [merge_transition_rich_periodic_support_refresh_shards.sh](/home/mila/l/lia/skae/scripts/merge_transition_rich_periodic_support_refresh_shards.sh).
  Compute-node smoke job `9361455` completed with `32` ok rows and `0`
  failures under
  [results/periodic_support_refresh_smoke_20260425_cal_square](/home/mila/l/lia/skae/results/periodic_support_refresh_smoke_20260425_cal_square).
  Full fixed-`17` seed-`0` LISTA-only shards `9361464` and `9361465` completed
  cleanly under
  [results/periodic_support_refresh_fixed17_seed0_20260425](/home/mila/l/lia/skae/results/periodic_support_refresh_fixed17_seed0_20260425).
  Dense LISTA completed `16/16` specs with `34,440` rows (`34,176` ok,
  `264` skipped, `0` failures), and blockdiag LISTA completed `17/17` specs
  with `38,280` rows (`38,016` ok, `264` skipped, `0` failures). Merge job
  `9361470` is still pending on scheduler priority, but the per-root summaries
  already provide the science read. The decisive positive result is dense
  LISTA exact `topk:8` after the trajectory is clearly in the target basin:
  refreshed-support routing reaches route-target fraction `0.8552/0.8886`,
  fallback `0.1392/0.1058`, and
  refreshed-versus-previous-support MSE ratio `0.0093/0.0131`. Dense LISTA
  `topk:8` family and blockdiag `topk:8` family also support the mechanism,
  but blockdiag exact supports do not. Therefore the stronger mechanism claim
  should be written for dense LISTA exact `topk:8` and for support families,
  not as a universal statement about every LISTA support definition.
- The supporting Dysts long-horizon visual packet for senior-coauthor handoff
  is now complete at
  [H5000](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h5000_lista_phase_portraits_manifest.json),
  [H20000](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h20000_lista_phase_portraits_manifest.json),
  the older LISTA-only shared-batch
  [H30000](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h30000_lista_phase_portraits_manifest.json),
  and the benchmark-aligned all-roots
  [H30000 best-root packet](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h30000_best_root_phase_portraits_manifest.json).
  The new all-roots `H30000` selector uses the completed seven-root collector
  rows and picks the lowest `H30000` best-periodic MSE per system across all
  checked-in roots, seeds, and periodic cadences; it selects block-diagonal
  LISTA on `14/15` systems (`sc=6e-3` on `10`, `sc=3e-3` on `4`) and dense
  LISTA only on `dysts:QiChen`. Use that as the preferred `H30000` visual
  appendix. The older dense-heavy `H30000` shared-batch LISTA packet remains
  useful only as a selector-sensitivity artifact; it does not replace the
  root-level aggregate result that still favors block-diagonal LISTA
  (`sc=6e-3`) at `H30000`.
- The seed-`10` Dysts long-horizon benchmark reevaluation packet under
  [results/dysts_long_horizon_eval_20260414](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414)
  is now complete. The refreshed collector summary
  [summary.md](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/collect/summary.md)
  reports `750/750` complete tasks and `0/750` pending or invalid tasks.
- The cache-length infrastructure bug is fixed and no longer the blocker.
  Cache prebuild `9273655`, validation `9273656`, collector `9273658`, and
  replacement validation probe `9273675` all completed cleanly, so the only
  remaining issue at that point was rescue coverage on the failed `9273657`
  array tasks.
- The concrete rescue blocker is now identified and fixed in
  [skae/config.py](/home/mila/l/lia/skae/skae/config.py): older checkpoints
  serialize legacy environment fields such as `ENV.COMPETITIVE_LV.SYSTEM_SEED`,
  and the prior `Config.from_dict()` path rejected those unknown keys before
  evaluation began. A previously failing zero-sparse MLP reevaluation
  (`dysts:Chua`, seed `0`) now completes cleanly after that compatibility
  patch.
- Rescue pass `1` is complete. `9278881` and recollect `9278882` both
  finished cleanly, and the queue metadata remains in
  [rescue_pass1_queue_record.json](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414/queue/rescue_pass1_queue_record.json).
- The Dysts long-horizon benchmark is now complete and decision-grade as
  supporting evidence: dense LISTA is the best aggregate benchmark root at
  `H5000/H10000` (`0.1285/0.9778`), while block-diagonal LISTA `sc=6e-3` is
  best at `H20000/H30000` (`1.9150/2.2720`). Both MLP controls are now fully
  measured too: sparse MLP `0.1953/1.2373/3.2524/3.6981` and zero-sparse MLP
  `0.2474/1.4564/3.2354/3.7893`.
- The paper Dysts launcher now supports the missing block-diagonal MLP family
  directly. `generic_sparse_blockdiag` is now a paper benchmark variant in
  [skae/benchmarks/paper_benchmark_manifest.py](/home/mila/l/lia/skae/skae/benchmarks/paper_benchmark_manifest.py),
  and
  [scripts/queue_paper_followup_recipes.sh](/home/mila/l/lia/skae/scripts/queue_paper_followup_recipes.sh)
  no longer fails if the historical selected-`dt` table is absent; it falls
  back to benchmark-default `dt` values and keeps the comparison anchors
  stable for partial reruns.
- The Dysts block-diagonal MLP extension is now complete at the paper
  collector horizons under
  [results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415).
  Replacement wrapper `9282328`, collector `9282331`, and compare jobs
  `9282332-9282334` all finished, and replacement training array `9282330`
  ended with `299/300` successful tasks. The lone historical miss was
  `generic_sparse_blockdiag_ns200k_sc3em3` on `dysts:Dadras`, seed `0`
  (`9282330_150`), which failed with a CUDA uncorrectable ECC error on a
  Quadro RTX 8000 rather than a model-side crash; retry `9286093_150` later
  cleared that gap and enabled the full long-horizon packet below. On the Dysts-only
  `H100/H500/H1000` packet, system-median best-periodic MSE is
  `9.227e-05 / 0.001412 / 0.004684` for `sc=3e-3` and
  `7.454e-05 / 0.001399 / 0.004821` for `sc=6e-3`; among the two new roots,
  `sc=6e-3` is slightly better at `H100/H500` while `sc=3e-3` is slightly
  better at `H1000`.
- The matching long-horizon custom-root Dysts packet is now also complete under
  [results/dysts_long_horizon_eval_mlp_blockdiag_20260415](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_mlp_blockdiag_20260415).
  Retry `9286093_150` cleared the earlier hardware-transient `dysts:Dadras`
  gap, launcher `9286094` completed, and the chained jobs `9289755-9289758`
  finished cleanly. The collector
  [summary.md](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_mlp_blockdiag_20260415/collect/summary.md)
  reports `300/300` complete tasks and `0` pending tasks at
  `H5000/H10000/H20000/H30000`.
- The seven-root Dysts architecture audit is now complete. Aggregate median
  best-periodic MSE remains dense LISTA `0.1285/0.9778` at
  `H5000/H10000` and blockdiag LISTA `sc=6e-3` `1.9150/2.2720` at
  `H20000/H30000`. The new blockdiag-MLP roots land at
  `0.1501/1.1401/3.0536/3.5891` (`sc=3e-3`) and
  `0.1945/1.2761/2.9519/3.4785` (`sc=6e-3`), so they sharpen the fairness read
  but do not overturn the long-horizon headline. The strongest blockdiag-MLP
  root is the runner-up aggregate model at `H5000/H10000`, while neither
  blockdiag-MLP root wins any systems at `H20000/H30000`.
- The fixed-`17` matched hard-init MLP control follow-up is no longer just
  queued. Under
  [results/transition_rich_hardinit_mlp_controls_seed10_20260416](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416),
  launcher `9285895`, initial array `9285897`, pass-`0` collect / resolve /
  advance `9285898 -> 9285899 -> 9285900`, and rescue pass `1`
  `9291399 -> 9291400 -> 9291401 -> 9291402` have all finished. Pass-`1`
  [dt_resolution/pass1/dt_resolution.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/dt_resolution/pass1/dt_resolution.md)
  shows all `51/51` arm-system pairs as `accepted_default`, so the rescue /
  `dt` blocker is closed. The finalized forecasting summary under
  [collect_pass1/forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/collect_pass1/forecasting_summary.md)
  reports system-median best-periodic
  `H100/H500/H1000 = 0.0082 / 0.0260 / 0.0273` for the sparse hard-init MLP,
  `0.0094 / 0.0359 / 0.0383` for the structured blockdiag hard-init MLP, and
  `0.5704 / 2.6733 / 3.8044` for the tanh / no-shrink hard-init control. That
  strengthens the induced-sparsity story while weakening any architecture-only
  hard-init forecasting claim.
- The replacement hard-init interpretability chain is now complete after a
  small merge-reader patch that normalizes empty CSV cells. Shards
  `9304602-9304604` had already finished cleanly; the patched merge reran as
  `9304747` and the dependent summary as `9304748`, producing
  [interpretability_final_pass1](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/interpretability_final_pass1)
  with `13,554` rows and `0` failures plus the finalized matched-sampling
  comparison in
  [final_comparison_pass1](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1).
  On the paper slice (`absolute:0.001` / `deep`), the two sparse hard-init MLP
  controls are almost tied: blockdiag sparse MLP gives
  `0.0082 / 0.0252 / 0.0264` at `H100/H500/H1000`, plain sparse MLP gives
  `0.0082 / 0.0260 / 0.0273`, both have `H(B|S)=0.0000`,
  `H(S|B)=0.2068`, `U_exact ~= 0.98`, and `H(F|B)=0.0000`, while the
  tanh / no-shrink control remains far worse. That closes the matched
  hard-init fairness table and says induced sparsity matters more than the
  specific sparse-encoder architecture in this oversampled setting.
- The new operator-selection mechanism package is also complete under
  [results/transition_rich_operator_selection_hardinit_matched_20260418](/home/mila/l/lia/skae/results/transition_rich_operator_selection_hardinit_matched_20260418).
  Smoke `9304650`, shards `9304655-9304659`, and merge `9304660` all finished
  cleanly, and the merged packet writes `56,538` rows with `0` failures. This
  is the first direct held-out `A_global` versus `A_basin` versus
  `A_support/family/group` study with count-matched random controls,
  latent-kmeans controls, and masked-`K` projections on the matched hard-init
  root family. Its headline result is negative for the strong paper claim:
  even oracle basin-conditioned fits do not beat one global latent law on the
  fixed `17` systems, and the best support-family fits beat random controls
  but still remain worse than global. The execution blocker is therefore
  closed; the remaining blocker is claim positioning.
- The reframed centered-chart mechanism packet is now complete under
  [results/transition_rich_centered_chart_mechanism_20260420](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420).
  Shards `9310546-9310548` and merge `9310549` all finished cleanly, and the
  merged packet writes `74,369` rows with `0` failures. This reruns the
  mechanism question with centered local charts, depth strata, and the actual
  dense `tanh` / no-shrink MLP control. On `relative:0.1` exact support,
  `persistent_current`, and deep `q4` states, centered support-conditioned
  local slopes beat the learned global `K` on `93.1%` of evaluated blockdiag
  LISTA rows (`130`), `98.6%` of dense LISTA rows (`141`), and `100%` of the
  dense no-sparsity MLP rows (`140`). Deep support-gated `K` is also strongly
  positive, especially for blockdiag LISTA, where q4 input-gated/global-`K`
  and block-submatrix/global-`K` both win on `100%` of evaluated rows.
- The updated paper-safe reading is therefore more specific. The April 18
  zero-intercept operator-selection failure was mainly a charting/comparator
  problem, not evidence that local laws were absent. Once the read is done in
  centered local charts, support-, family-, and basin-conditioned local laws
  appear across much of the fixed `17`, including most of the proxy-labeled
  benchmark. But the dense no-sparsity MLP also shows the same deep centered
  local-law effect, so the paper still cannot claim that LISTA-style induced
  sparsity uniquely creates those local laws. The defensible claim is now:
  induced sparsity improves basin-support identifiability, stability, and
  forecasting; centered local-law evidence is real but not LISTA-specific; and
  the cleanest direct support-gated `K` interpretation is the blockdiag LISTA
  case. Boundary-adjacent `q1` states remain the main negative slice.
- The non-oracle self-routed forecasting packet is now complete rather than
  merely queued. It is implemented in
  [tools/evaluate_transition_rich_self_routed_forecasting.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_self_routed_forecasting.py)
  with shard / merge launchers in
  [scripts/run_transition_rich_self_routed_forecasting.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_self_routed_forecasting.sh),
  [scripts/merge_transition_rich_self_routed_forecasting_shards.sh](/home/mila/l/lia/skae/scripts/merge_transition_rich_self_routed_forecasting_shards.sh),
  and
  [scripts/queue_transition_rich_self_routed_forecasting_shards.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_self_routed_forecasting_shards.sh).
  Smoke validation is complete under
  [results/transition_rich_self_routed_forecasting_smoke_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_smoke_20260420)
  with `270` rows and `0` failures, and merge-path validation is complete
  under
  [results/transition_rich_self_routed_forecasting_merge_smoke_20260420/merged](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_merge_smoke_20260420/merged).
  This is the direct deployment-facing test of whether the model's own support
  or support family can route forecasting better than one global `K` without
  oracle basin labels. The full fixed-`17` packet is complete under
  [results/transition_rich_self_routed_forecasting_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420),
  with the original root-only queue `9314170-9314173`, the first coarse
  seed-split `long-cpu` queue `9314196-9314214`, the temporary `main-cpu`
  workaround `9314400-9314406`, and the `12h` long-queue retry
  `9314431-9314437` all canceled after scheduler inspection. The final
  successful queue was the one-seed `long-cpu` submission:
  `9314443-9314472` at `03:00:00`, plus dependency-held merge `9314473`.
  `sacct` now shows every shard and the merge as `COMPLETED 0:0`; shard
  elapsed times ranged from `00:39:51` to `02:08:48`, and the merge finished
  in `00:00:20`. The merged packet writes `510/510` completed runs,
  `24,600` rows, and `0` failures. The paper-facing Table 2 display now uses
  `H100/global` IQM ratios plus paired Wilcoxon/Holm counts because this is
  where the current packet gives the strongest confirmatory non-oracle routing
  evidence: dense LISTA exact-support `topk:8` routes clear `12/17` and
  `11/17` systems on the all slice and `14/17` and `15/17` on the deep slice.
  The older `H1000/global` medians / win rates remain useful descriptive
  long-horizon context (`0.228 / 0.920` for `support_gated_k` and
  `0.275 / 0.947` for `support_local_centered`), but the `H1000` paired
  Wilcoxon/Holm counts are underpowered until the seed expansion lands. The
  evaluator was also patched for resumable intra-shard reruns with atomic per-spec
  flushing; that applies to future reruns rather than retroactively to the
  completed queue. The resume path is compute-validated too: validation job
  `9315112` completed in `16s` and confirmed that rerunning the completed
  one-spec smoke shard skips work immediately with `1/1` completed.
- The fixed-`17` LISTA phase-portrait handoff packet is now complete at
  [H1000/H3000/H5000](/home/mila/l/lia/skae/docs/figures/fixed17_lista_phase_portraits_20260414/fixed17_h1000_h3000_h5000_lista_phase_portraits_manifest.json).
  It writes one figure per system per horizon (`51` total) plus per-system
  selection metadata, selecting each system's run by the lowest saved
  `H1000` best-periodic mean across collected transition-rich LISTA rows and
  then reusing that run's saved `H1000` best-periodic mode for all three
  horizons. The packet spans `8` distinct LISTA roots, so it should be read
  as supporting presentation material for coauthors and appendix use rather
  than as new causal shortlist evidence.
- The one-seed `17 x 2` LISTA basin-partition sweep at default `dt` was
  launched under
  [transition_rich_basin_partition_20260407](/network/scratch/l/lia/skae/transition_rich_basin_partition_20260407),
  with the `dt`-rescue continuation already dependency-chained behind it. That
  April 7 default-source queue used the older `200k` training budget and
  should now be read as a legacy long-budget LISTA root rather than the
  forward default. The submitted rescue task tables in that same chain were
  rewritten to `20k`.
- Forward interpretability queues should now use `20000` training steps and `1` seed while
  we are still diagnosing model behavior and choosing the right metrics; only
  the final locked paper rerun on this branch should return to `200000` and `10` seeds,
  except for the now-completed narrow default-sampling `v6` / `v7`
  forecast-floor follow-up, which did not reopen the shortlist.
- An April 13 audit of the last 7 days of paper-critical SLURM work found no
  finished packets missing from this file. The required cleanup is status
  hygiene: the clean tanh / no-shrink control and the `v7`-screened method
  axes are complete and should no longer be described as pending.
- The seed-`0` hard-init follow-up is now complete under
  [transition_rich_basin_partition_hardinit_seed0_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409),
  with chain `9209614 -> 9209615 -> 9209616` completed cleanly. Forecasting
  summary:
  [forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409/collect/forecasting_summary.md).
  Interpretability summary:
  [interpretability_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409/reduce/interpretability_summary.md).
- The hard-init read is useful but not uniform. The block-diagonal hard-init
  variant is the clearer interpretability positive: at `absolute:0.001` on
  deep-basin states it improves `H(S|B)` (`1.4297 -> 1.3493`), `U_exact`
  (`0.7181 -> 0.7447`), `H(F|B)` (`0.1129 -> 0.1018`), own-basin projection
  ratio (`25.5197 -> 7.7018`), and wrong-support ratio (`0.7599 -> 0.3034`)
  with nearly neutral forecasting. The dense `p64` hard-init variant is more
  mixed on raw support compression, but it improves forecasting strongly
  (`H1000` system-median best `0.1358 -> 0.0794`) and also reduces the
  damage from deep-basin canonical-support interventions
  (`own/base 9.9799 -> 3.0431`, `freeze/base 0.8926 -> 0.6768`).
- The first post-hard-init cross-root paper-facing reduction is now already
  launched under
  [transition_rich_post_hardinit_crossroot_eval_20260409](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409).
  Its phase-`1` dependency chain was `9210427` (`collect_tr_crossroot`,
  `afterok:9209616`) -> `9210429` (`tr_interp_crossroot`,
  `afterok:9210427`). Both jobs completed. Its selected comparison bundle is fixed in
  [selected_roots.txt](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409/root_specs/selected_roots.txt).
  That queued bundle is the minimal paper-facing comparison set: strongest
  current `v5` forecast-retaining and exact-support roots, strongest current
  `v6` identifiability follow-ups, matched `v1` sparse MLP control, and the
  hard-init packet's base / variant pairs.
- That first submitted cross-root phase-`1` reduction was not usable.
  `9210429` wrote `0` rows and `17` failures under
  [failures.json](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409/interpretability/failures.json).
  The log shows `ROOT_LABELS_CSV` arrived as only `v5_blockdiag_signsplit`,
  and every attempted row then failed checkpoint load because the saved
  block-diagonal LISTA checkpoints still expose `encoder.We.*` keys while the
  current loader expects the newer `precode_module` / `dict_param` layout.
  That run should be treated only as a failed first attempt, not as evidence.
- That clean rerun has now completed under
  [interpretability_rerun_compat_20260409](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409/interpretability_rerun_compat_20260409)
  as job `9211252`, with `4131` interpretability rows and `0` failures. The
  rerun validates the same shortlist choice already suggested by the hard-init
  packet: the forecast-retaining interpretability finalist is the
  block-diagonal hard-init root, the stronger forecasting / intervention
  tradeoff is the dense `p64` hard-init root, and the matched sparse MLP
  control does not beat either finalist on the main branch objective.
- The study-plan-aligned state-level reducer is now smoke-validated on the
  historical native trio under
  [transition_rich_interpretability_smoke_20260409/native_seed0](/home/mila/l/lia/skae/results/transition_rich_interpretability_smoke_20260409/native_seed0).
  Its current paper consequence is sharper than the earlier support-group
  reducer: deep-basin `H(B|S)` is already approximately zero for both dense
  and block-diagonal LISTA on that subset, but `H(S|B)` remains large unless
  supports are forced into aggressive top-`k` masks. The active positive is
  basin purity, not exact-support uniqueness.
- The next study-plan metric tier is now also smoke-validated on that same
  native trio under
  [transition_rich_interpretability_smoke_20260409/native_seed0_v2_family_projection](/home/mila/l/lia/skae/results/transition_rich_interpretability_smoke_20260409/native_seed0_v2_family_projection).
  It adds greedy support-family clustering plus deep-basin canonical-support
  projection diagnostics. Its current paper consequence is that the branch now
  has a plausible family-level positive even before the new ablations land:
  family entropy within basin is already much lower than exact-support entropy
  on the native trio, but the exact canonical-support intervention still hurts
  one-step prediction even when the wrong-basin intervention is catastrophic.
  That means the defensible near-term story is basin-selective support
  families or dominant groups, not yet one exact canonical support per basin.
- All submitted fixed-`17` interpretability packets `v1-v6` are now complete
  through arrays, collect, resolve, and interpretability reduction.
- The completed packet ladder now covers the full planned shortlist sweep on
  this branch: `v1` initial dense / blockdiag / MLP control, `v2`
  HyperLISTA and `2 x basin-count` blocks, `v3` reset-policy and dynamics-
  aware reencoding, `v4` structured and soft-block penalties, `v5` sign-split
  plus latent-size / block-count sweeps, and `v6` restrained pre-code plus
  decoder coherence.
- `v5` is the sign-split shortlist tier in concrete terms:
  hard block-diagonal and dense soft-block LISTA families with sign-split
  codes, `2` versus `4` LISTA refinement loops, doubled block-count variants,
  and latent-size sweeps including `p=64` and `p=128`.
- `v6` is the identifiability follow-up to `v5`:
  the same sign-split shortlist families with either a restrained linear
  pre-code, a decoder-coherence penalty, or both. Decoder coherence here
  means penalizing off-diagonal similarity among normalized decoder atoms so
  the decoder dictionary has fewer redundant atoms and fewer interchangeable
  exact supports.
- The best current shortlist exact-support result is the `v5` root
  `lista_dense_softblock_signsplit_p64_basin_partition`. On deep-basin states
  at `absolute:0.001`, it reaches `mean H(S|B)=0.7719`,
  `mean U_exact=0.8064`, and `mean H(F|B)=0.0521` while still keeping
  `16/17` systems under the `H1000` good-forecast gate. This is the first
  branch result that compresses exact supports within basin strongly enough to
  matter without collapsing forecasting outright. It is also worth considering `v6` as well.
- The best forecast-retention result across the new shortlist tiers remains
  the `v5` root `lista_blockdiag_signsplit_basin_partition`
  (`H1000 system-median = 0.0119`, `17/17` good systems). The strongest `v6`
  forecast-preserving root is
  `lista_dense_softblock_signsplit_coherence_basin_partition`
  (`H1000 system-median = 0.0585`, `17/17` good systems).
- `v6` resolves the next mechanism tradeoff but does not displace `v5`.
  Coherence-only roots help forecasting; linear-encoder roots help exact-
  support compression; their combination still leaves a large forecasting cost
  and does not beat the `v5 p64` root on the combined frontier. That being said, we should consider `v6` as well.
- The next design-note LISTA tier is now implemented locally as well:
  standard LISTA supports adaptive residual/prior-gap thresholds, groupwise
  base thresholds over inferred latent groups, and explicit free-MLP,
  dictionary-tied, or hybrid tied-plus-residual pre-codes. The fixed-`17`
  transition-rich manifest now has runnable shortlist variants for those
  axes. The block-diagonal adaptive/groupwise-threshold arm plus the dense
  dictionary-tied and hybrid pre-code arms have now all been screened in the
  completed `v7` packet, so they no longer sit in the unrun backlog; they are
  now negative / mixed shortlist provenance.
- The next still-open LISTA design-note axis is now implemented locally too:
  standard LISTA supports fixed-beta momentum refinement, and the fixed-`17`
  transition-rich manifest now includes runnable sign-split momentum variants
  on both the forecast-retaining block-diagonal root and the dense soft-block
  `p64` root. Those runs were also screened in `v7` and did not beat the
  locked finalists, so momentum is no longer a tooling blocker or an unrun
  shortlist gap.
- The stronger soft-block sweep that was still open in the design note is now
  exposed locally too: the fixed-`17` transition-rich manifest now includes
  sign-split dense soft-block `p64` shortlist variants at `5e-4` and `1e-3`
  in addition to the earlier `1e-4` weight. Those runs have now been
  screened in the completed `v7` packet, so the open question is now
  evidence rather than configuration coverage.
- A narrow hard-init follow-up is now complete under
  [transition_rich_basin_partition_hardinit_seed0_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_hardinit_seed0_20260409).
  Its chain `9209614` (training array) -> `9209615` (`collect_tr_bp`) ->
  `9209616` (`tr_interp_reduce`) finished cleanly. This packet was the first
  forward execution of the near-separatrix hard-initialization axis on the
  fixed shortlist. It kept the working `20k` budget, used seed `0` only, and
  compared four roots: the current `v5` block-diagonal and dense soft-block
  `p64` anchors against their corresponding hard-init variants.
- The matched sparse MLP control and the clean tanh / no-shrink `200k`,
  `10`-seed control are now both in hand, so the main missing paper-side step
  is no longer training-side queue completion. It is the state-conditioned
  evaluation read: reduce the locked finalists and controls on the study-plan
  metrics by basin depth / separatrix proximity and decide whether any method
  tie-break remains worth budget.
- One narrow default-sampling LISTA-only refinement around the `v5 p64`
  recipe has now been run. The highest-value remaining method-side axes from
  the design notes included stronger soft-block penalties, momentum LISTA,
  adaptive or blockwise thresholds, dictionary-tied or hybrid pre-codes,
  and group-aware sparse-group shrinkage / top-`k` group-first support
  selection. Those variants are now implemented and at least smoke-screened on
  the fixed `17` systems, so they no longer belong to the “unrun” list.
- That narrow follow-up is now complete under
  [transition_rich_basin_partition_20260410_seed0_smoke_v7](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260410_seed0_smoke_v7)
  as wrapper `9226563`. It keeps the working `20k`, seed-`0` budget and
  queues only unrun shortlist variants around the current winners:
  block-diagonal adaptive/groupwise thresholds, block-diagonal sparse-group
  shrinkage, block-diagonal sign-split momentum, dense soft-block
  dictionary-tied pre-code, dense soft-block hybrid pre-code, denser
  soft-block penalties at `5e-4` and `1e-3`, and dense soft-block sign-split
  momentum. The chain `9226564_[0-135] -> 9226565 -> 9226566 -> 9226567` is
  now complete with `0` reducer failures. Its task table leaves
  `hard_init_oversample` unset, so this packet stays on the default sampling
  regime and remains directly comparable to the MLP controls. The best new
  forecasting root is `lista_blockdiag_sparsegroup_basin_partition`
  (`H1000 system-median best = 0.0846`), while the strongest new deep-basin
  support-compression read is
  `lista_dense_softblock_signsplit_p64_softblock5em4_basin_partition`
  (`H(S|B)=0.6795`, `U_exact=0.8453`, `H(F|B)=0.0634` at
  `absolute:0.001` / `deep`), but that root is missing `1/17` systems after a
  fast training failure. `lista_blockdiag_adaptive_groupwise_threshold` is not
  a serious contender because it is numerically unstable. No new `v7` root
  cleanly dominates both forecasting and support alignment, so treat this wave
  as shortlist provenance rather than as a promotion over the locked
  finalists. The completed hard-init packet remains the separate
  sampling-axis ablation and the main promoted comparator row.
- One narrow default-sampling `200k`, `10`-seed forecasting-only follow-up is
  now complete under
  [transition_rich_basin_partition_v6_v7_200k_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_basin_partition_v6_v7_200k_seed10_20260410).
  It promotes `lista_dense_softblock_signsplit_coherence_basin_partition` and
  `lista_blockdiag_sparsegroup_basin_partition`, the best forecasting roots
  from `v6` and `v7`, to test whether longer training lowers their
  default-sampling forecast floor enough to reopen the shortlist. Wrapper
  `9228393` wrote the `340`-task default table, but its attempted model-wise
  dt-rescue continuation was canceled because that protocol would allow
  different models on the same system to run at different `dt` values and
  would therefore confound the forecasting comparison. The actual fair queue,
  `9228394_[0-339] -> 9228395`, completed on April 11, 2026 and wrote
  `329/340` forecasting rows across both roots and all `17` systems. The
  better root, `lista_dense_softblock_signsplit_coherence_basin_partition`,
  reaches system-median best-periodic `H100/H500/H1000 = 0.0416 / 0.0761 / 0.0796`;
  `lista_blockdiag_sparsegroup_basin_partition` lands at
  `0.0437 / 0.1142 / 0.1193`. This is not a full branch reopen; it is a
  completed long-budget forecasting floor check at matched system-level `dt`,
  and it is negative for promotion. The coherence root remains slightly worse
  than the promoted dense hard-init finalist at all three horizons
  (`0.0196 / 0.0733 / 0.0775`) and worse than the matched sparse MLP control
  at `H500/H1000` (`0.0614 / 0.0608`). Because this packet is forecasting-
  only, it adds no new basin-support reduction. Return the remaining branch
  budget to state-conditioned evaluation rather than more training.
- The main remaining evaluation-side gaps from
  [docs/planning/interpretability_study_plan.md](/home/mila/l/lia/skae/docs/planning/interpretability_study_plan.md)
  have narrowed after the corrected reviewer-response reruns. Controlled-
  transfer switch-timing diagnostics now give a useful dense LISTA exact
  `topk:8` support-switching read, while true Jacobian/eigendirection
  diagnostics are mixed and should remain secondary. Remaining paper-side work
  is mainly merge/artifact finalization for the completed
  periodic-support-refresh/no-refresh ablation, final state-conditioned
  long-horizon forecasting figures at
  `H100/H500/H1000` by basin depth / separatrix proximity, seed/threshold
  robustness only if the transfer branch becomes main-text evidence, and
  basis-aware or similarity-aligned operator-family wording if we go beyond
  the current support-routing claim.
- The reducer-side tooling for those evaluation items now exists locally:
  canonical wrong-support rollout metrics, first-switch timing summaries,
  sampled effective-Jacobian family summaries, and optional visual artifacts
  for support families are implemented in the state-level interpretability
  reducer. The remaining gap is no longer missing code; it is running those
  diagnostics on the completed shortlist roots and deciding how they change
  the final claim. One caveat after the April 9 basis-aware update: the live
  reducer already supports the **raw** operator-family and Jacobian-family
  summaries, but it does **not** yet implement the new similarity-aligned
  operator distances, eigendirection comparisons, or invariant-subspace-angle
  metrics. Those alignment-aware diagnostics remain a tooling gap.
- The overnight dependency chain behind that root is now effectively complete
  through pass `6`, and its scientific consequence is stronger default-`dt`
  evidence rather than smaller-`dt` rescue evidence:
  [collect_pass0](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/collect_pass0/forecasting_summary.md)
  already reports `17/17` good systems at `H1000` for both dense and
  block-diagonal LISTA on the fixed shortlist, and the rescue-only `20k`
  reruns keep all `16/16` rerun systems below the same gate by system median
  for both roots.
- The same overnight chain also exposed a queueing / collection bug rather
  than a scientific need for more halving: the collector preserved `env_dt`
  only for `multiwell_strong_transition`, so the resolver kept re-emitting
  default-`dt` task tables for `gated_*` and `claude:*` arms. That collector
  gap is now fixed locally in
  [collect_forecasting_roots.py](/home/mila/l/lia/skae/tools/collect_forecasting_roots.py).
  Do not interpret the completed overnight waves as evidence for a
  smaller-`dt` effect.
- Natural live-sweep `evaluation_best/.../rollout_artifacts.pt` outputs have
  not appeared yet; the current native-trio read comes from manual compute-node
  reevaluation of saved checkpoints under
  [manual_eval](/network/scratch/l/lia/skae/transition_rich_basin_partition_20260407/manual_eval).
- The first manual native-system read from that live sweep already says the
  native trio is not bottlenecked by default `dt` under the current user-facing
  rescue rule, and it already gives the branch's key positive signal:
  support groups remain basin-pure in every inspected dense / block-diagonal
  LISTA arm.
- The current lead blocker is therefore no longer queue completion,
  step-size rescue, or missing implementation support. The locked `200k`,
  `10`-seed LISTA-vs-control packet is now fully reduced, but it is still a
  mixed-regime comparison because the promoted LISTA roots use hard-init
  oversampling while the completed MLP controls in that packet use the
  standard training sampling regime. The matched-sampling hard-init control
  packet under
  [results/transition_rich_hardinit_mlp_controls_seed10_20260416](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416)
  is now forecasting-finalized and says the hard-init forecasting gain is not
  LISTA-exclusive: the sparse hard-init MLP control reaches
  `H100/H500/H1000 = 0.0082 / 0.0260 / 0.0273`, the structured blockdiag
  hard-init MLP reaches `0.0094 / 0.0359 / 0.0383`, and both beat the locked
  hard-init LISTA forecasting winner at `H1000` (`0.0516`), while the clean
  zero-sparse hard-init control is much worse. The replacement sharded
  interpretability rerun is also complete, so the remaining blocker is no
  longer missing reduction output. The remaining claim-calibration work is to
  keep the sampling-regime split explicit, decide how prominently to keep the
  block-diagonal hard-init forecast-retaining companion result, and decide
  whether the final wording should stay at exact-support reuse or soften to
  family / dominant-group or symmetry-aware alignment in light of the now-
  completed centered-chart mechanism packet and the now-completed non-oracle
  self-routed / state-conditioned
  `H100/H500/H1000` forecasting read.
- Immediate wrap-up priority should therefore be:
  first update the coauthor-facing docs so the locked hard-init packet is read
  as mixed-regime evidence rather than architecture-only evidence; second
  finish or freeze the remaining matched hard-init MLP control coverage and
  then prepare the paper-facing figures / tables with an explicit
  sampling-regime split; third run only the extra diagnostics still needed for
  wording, chiefly the basis-aware alignment readouts and the final visual
  summaries built from the completed self-routed long-horizon forecast read.
- The concrete post-hard-init evaluation bundle is now complete in phase `1`
  form under
  [transition_rich_post_hardinit_crossroot_eval_20260409](/home/mila/l/lia/skae/results/transition_rich_post_hardinit_crossroot_eval_20260409).
  It compares `v5` forecast-retaining and exact-support leaders, `v6`
  coherence and restrained-linear follow-ups, the matched non-LISTA control,
  and the hard-init packet's near-separatrix oversampling variants under one
  reduction protocol.
- Rationale for that exact bundle:
  it is the smallest set that can answer the paper-facing causal question
  without spending more training budget first. That bundle has now done its
  job: it fixed the seed-`0` shortlist before the locked multi-seed rerun, and
  the locked packet then answered the causal comparison at the final budget.
- The branch now has the fully reduced final confirmatory rerun in hand. The
  seed-`10` packet is
  [transition_rich_basin_partition_final_seed10_20260409](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409):
  default array `9211290_[0-679]`, pass-`0` collect / resolve
  `9211291 -> 9211292`, rescue pass-`1` array `9214917_*`, pass-`1`
  collect / resolve `9214918 -> 9214919`, final reducer `9218036`, and final
  paired summary `9218037`.
  It runs the promoted LISTA roots
  `lista_blockdiag_signsplit_hardinit_basin_partition` and
  `lista_dense_softblock_signsplit_p64_hardinit_basin_partition` against both
  `mlp_sparse_basin_partition_control` and the newly exposed
  `mlp_zero_sparse_basin_partition_control`, which in that locked packet
  should now be read as a zero-`L1` ReLU ablation rather than the clean tanh /
  no-shrink baseline, at the locked `200k`, `10`-seed budget on the fixed
  `17` systems.
- Because those promoted LISTA roots are hard-init oversampled while the two
  MLP controls are standard-sampling, this packet should be read as mixed
  architecture-plus-sampling evidence. It is useful for ranking and for
  motivating a matched-regime follow-up, but it is not the final
  architecture-isolating comparison.
- The early failures in that array are operational rather than scientific:
  tasks `238`, `255`, `267`, `279`, `295`, and `303` all failed on `cn-a009`
  with `torch.AcceleratorError: CUDA error: uncorrectable ECC error
  encountered`.
  Rescue pass `1` resolved that hardware-only cluster; no further rescue rows
  were requested afterward.
- The finalized forecasting summary under
  [collect_pass1/forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/collect_pass1/forecasting_summary.md)
  reports system-median best-periodic `H1000` values of `0.0516` for the
  block-diagonal hard-init LISTA finalist, `0.0775` for the dense soft-block
  `p=64` hard-init LISTA finalist, `0.0608` for the matched sparse MLP
  control, and `0.0909` for the locked packet's zero-`L1` ReLU MLP
  control, with all four roots at `17/17` good systems.
- That zero-`L1` ReLU arm should not be treated as disposable. It is not the
  clean tanh / no-shrink anti-sparsity control, but it is still
  scientifically meaningful because it removes the explicit `L1` penalty while
  retaining ReLU-induced architectural shrinkage. In other words, it is a
  ReLU-only sparsity ablation that can still speak to basin-support
  identification and forecasting quality.
- That locked forecasting summary is useful but not sufficient for the
  coauthor-facing mechanism writeup on this branch. The next paper-facing read
  should expand it to `H100`, `H500`, and `H1000`, then stratify those
  horizons by basin depth / separatrix proximity. The explicit prediction is
  that sparsity-driven gains should be largest deep in a basin and smallest
  near a separatrix if the latent is really preserving basin identity.
- The detailed fixed-`17` per-system forecasting table for the four locked
  roots now lives in
  [docs/EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md).
  By lowest per-system `H1000`, the block-diagonal hard-init LISTA root is
  best on `7/17` systems, the dense `p64` hard-init LISTA root is best on
  `5/17`, the sparse MLP control is best on `5/17`, and the zero-`L1`
  ReLU MLP control is best on `0/17`.
- A new working-budget zero-sparsity no-shrink control screen is now also
  complete under
  [transition_rich_zero_sparse_control_noshrink_20k_seed3_20260410](/home/mila/l/lia/skae/results/transition_rich_zero_sparse_control_noshrink_20k_seed3_20260410)
  as `9223056_[0-50] -> 9223057 -> 9223058`. It keeps all `17/17` systems
  inside the default-`dt` `H1000 < 50` gate, but its system-median
  best-periodic forecasting is weak (`H100/H500/H1000 = 1.8317 / 3.5797 /
  4.1857`), so it strengthens the sparse-vs-zero-sparse framing without
  changing the branch ranking.
- The older locked-budget zero-sparsity control expansion under
  [transition_rich_zero_sparse_control_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_zero_sparse_control_seed10_20260410)
  as array `9221521_*` is misconfigured relative to the requested clean
  tanh / no-shrink control because its task table maps
  `mlp_zero_sparse_basin_partition_control` to `generic_sparse` rather than
  `generic_no_shrink`. But it should still remain in the paper-side evidence
  stack as a ReLU-only ablation, not be thrown away outright. Operationally it
  now runs through collect / resolve pass `4`, still accepts default `dt` on
  all `17/17` systems, and remains at system-median best-periodic
  `H100/H500/H1000 = 0.5764 / 2.0556 / 2.6532` with no requested smaller
  `dt`.
- The corrected locked-budget tanh / no-shrink control is now complete under
  [transition_rich_zero_sparse_tanh_control_seed10_20260410](/home/mila/l/lia/skae/results/transition_rich_zero_sparse_tanh_control_seed10_20260410)
  as wrapper `9224111`, using a fresh results tag so the task table is rebuilt
  from the corrected manifest mapping. That launcher has now completed and the
  fresh task table is verified to use `config_name=generic_no_shrink`. The
  default pass now completes as `9224263_* -> 9224264 -> 9224265`; despite one
  fast array failure, the collector writes `169` rows, all `17/17` systems
  accept default `dt`, and the packet reports system-median best-periodic
  `H100/H500/H1000 = 0.5763 / 1.7924 / 2.4279`. This is the clean no-shrink
  control. It is slightly stronger than the ReLU-only zero-`L1` ablation but
  remains much weaker than the locked sparse LISTA roots, which strengthens
  the broader induced-sparsity claim.
- The final paired state-level comparison under
  [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md)
  is now the branch's main mixed-regime read. On the selected `absolute:0.001`
  / `deep` slice, the dense `p64` hard-init LISTA root beats the
  standard-sampling sparse MLP control in `H(S|B)` (`0.2449 -> 0.0543`), `U_exact`
  (`0.9772 -> 0.9923`), and `freeze/base@20` (`0.3923 -> 0.1691`), with
  paired wins on `15/17`, `14/17`, and `16/17` systems while remaining
  forecast-competitive. The block-diagonal hard-init root instead carries the
  best finalized locked-packet forecasting value (`H1000 = 0.0516`) plus
  better freeze robustness than both standard-sampling MLP controls, but it
  loses exact-support compression to the matched sparse MLP control and
  therefore should be written as the forecast-retaining companion rather than
  the lead basin-support win.
- The remaining paper-critical evidence gap is therefore still partly the
  LISTA-versus-MLP comparison: the matched-sampling hard-init forecasting read
  is now complete and already says sparse hard-init MLPs can inherit the
  hard-init forecasting gain, while the structured blockdiag hard-init MLP is
  also competitive. The stronger induced-sparsity claim is no longer blocked
  on training-side queue completion: the clean tanh / no-shrink `200k`,
  `10`-seed control is in hand, and the hard-init control packet now has full
  pass-`1` forecasting coverage plus all `51/51` arm-system pairs accepted at
  the default `dt`. The remaining evaluation-side gap is now the missing
  matched-hard-init state-level interpretability reduction, plus the planned
  `H100/H500/H1000` depth-versus-separatrix read needed to keep the fairness
  language precise.
- On that same selected slice `H(F|B)` is `0.0000` for all four roots, so the
  locked-packet discrimination is no longer family entropy. It is exact-
  support fragmentation, intervention stability, persistence, and forecasting.
- Mila rejected the fully pre-expanded rescue chain for that packet under
  `AssocMaxSubmitJobLimit`, so the live confirmatory execution now uses an
  incremental queueing pattern instead of a fully chained one: run the default
  array first, inspect resolve output, and only then submit the specific
  rescue pass that is actually needed. The branch now has a dedicated one-pass
  launcher for that path at
  [queue_transition_rich_basin_partition_rescue_pass.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_basin_partition_rescue_pass.sh).
- The final paired paper readout is also prepared locally at
  [summarize_transition_rich_final_comparison.py](/home/mila/l/lia/skae/tools/summarize_transition_rich_final_comparison.py),
  and the branch has now used it to emit one combined LISTA-vs-control readout
  under
  [final_comparison_pass1](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1)
  instead of another manual aggregation pass.
- The same packet now also has an auto-advance watcher at
  [advance_transition_rich_basin_partition_packet.sh](/home/mila/l/lia/skae/scripts/advance_transition_rich_basin_partition_packet.sh):
  job `9211747` completed after `9211292` and emitted the pass-`1` rescue
  array `9214917_*`. That same path then queued the final interpretability
  reduction and paper-facing comparison automatically and finalized at pass
  `1`; the recorded status is in
  [advance_pass1.json](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/automation/advance_pass1.json).
- Run that bundle in two phases:
  phase `1` is already queued with the existing reducer on raw support-family /
  projection / operator-family / Jacobian-family metrics; phase `2` remains a
  tooling task and should rerun the same bundle only after the reducer gains
  the April 9 basis-aware alignment readouts (similarity-aligned operator
  distances, eigendirection similarity, and symmetry-aware support alignment).
- Step-size rescue should therefore no longer be written as an open blocker on
  the one-seed LISTA shortlist. The current queue-era evidence already says the
  fixed `17`-system LISTA packet is operationally fine at default `dt`; what
  remains missing is the matched standard-MLP contrast and any follow-up
  ablation needed to explain why high-purity recurring supports still fail the
  stronger local-linearity test.
- Operational queue caveat:
  `v4` still carries a launcher-level failure record because `9202903` hit the
  submit cap, but all emitted default / rescue / reducer jobs completed
  successfully. There is no live backlog now, so any next packet can be
  submitted deliberately rather than under queue-pressure triage.
- Do not write this branch as if it needs to prove the stronger
  chart-switch-localization claim or a sparse-only mechanism claim. Those
  older questions are supporting context; the paper decision for this branch is
  whether LISTA gives cleaner basin-separated support structure than the
  standard MLP control on the fixed shortlist.
- The current ablation-design source of truth for the next interpretability
  loop is
  [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md).
  It should guide matrix-structure, reset, loss, and diagnostic sweeps for
  plan items `3` and `4` until those axes are executed and written back into
  the live docs.
- The forward interpretability scope is now frozen to `17` systems and no
  others:
  - native transition-rich trio:
    `multiwell_strong_transition`, `gated_local_linear`,
    `gated_transfer_linear`
  - Claude-catalog subset:
    `arrested_spiral`, `cal_asymmetric_3`, `cal_high_cross_3`,
    `cal_hexagon_6`, `cal_octagon_8`, `cal_pentagon_5`, `cal_square_4`,
    `checkerboard_potential`, `duffing_triple_well`, `snic_multi`,
    `transition_routes_4`, `var_depth_gradient_4`, `var_diamond_4`,
    `var_l_shape_5`
- The conceptual inventory in
  [docs/planning/transition_rich_system_inventory_20260406.md](/home/mila/l/lia/skae/docs/planning/transition_rich_system_inventory_20260406.md),
  the elite sketches in
  [docs/planning/transition_rich_elite_system_sketches_20260406.md](/home/mila/l/lia/skae/docs/planning/transition_rich_elite_system_sketches_20260406.md),
  and the companion figures under
  [docs/figures/transition_rich_inventory_20260406](/home/mila/l/lia/skae/docs/figures/transition_rich_inventory_20260406)
  now serve as historical design-space provenance only. Do not use their
  broader `16`-system / `8`-system shortlists as the live experiment scope.
- The worktree now also has an artifact-backed audit of the already
  implemented Claude catalog in
  [docs/planning/claude_catalog_audit_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_audit_20260407.md)
  with a companion figure under
  [docs/figures/claude_catalog_audit_20260407](/home/mila/l/lia/skae/docs/figures/claude_catalog_audit_20260407):
  - `112` systems are registered in `skae/claude_catalog`
  - the combined grounded screen now covers `83`
  - `29` implemented systems remain unscreened
  - `12` screened systems are now accepted under the fast-screen gates, with an
    `8`-system strict-crossing core
  - `var_depth_gradient_4` is now part of that strict core and `cal_hexagon_6`
    is now part of the relaxed-accept subset after targeted retune refreshes,
    while `cal_octagon_8` remains a screened high-basin near miss
  - the companion packet now includes a combined audit atlas plus separate
    strict-crossing and accepted-pass portrait galleries
  - the implemented Claude catalog should therefore be treated as a grounded
    small benchmark pool plus retune frontier rather than as an already
    validated large benchmark packet
  - use that audit to describe what is grounded and what remains frontier
  - do not treat the older `6`-system packet in
    [docs/planning/claude_catalog_handoff_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_handoff_20260407.md)
    as the active recommendation; forward Claude-catalog experiments are now
    restricted to the fixed `14`-system subset above
  - the same supersession is now also written in senior-coauthor-facing
    language in
    [docs/planning/claude_catalog_senior_review_packet_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_senior_review_packet_20260407.md),
    so the protocol and active scope are readable without relying on internal
    code names
- The first frozen `2D` pair remains useful, but it is now understood as the
  secondary half of a three-system suite:
  - `gated_local_linear` is now the cleanest oracle-validated chart-switching
    toy
  - `multiwell_strong_transition` is the weaker shared-corridor toy
  - `gated_transfer_linear` is the explicit-transfer stress test where periodic
    decode/re-encode may still help at chart changes, but the oracle state-space
    gap is only modest
- The first implementation tasks are complete locally:
  - write tests
  - add the minimal interface scaffolding
  - calibrate deterministic toy systems
  - freeze two native-plot secondary `2D` candidates
  - implement and calibrate the explicit-transfer toy
- The paper-facing 100-trajectory figure now exists at [transition_rich_100_trajectories.svg](/home/mila/l/lia/skae/docs/figures/transition_rich_20260331/transition_rich_100_trajectories.svg).
- The transfer-system figure packet now also exists under
  [docs/figures/chart_switching_transfer_20260331](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331).
- Short training-entry smokes also pass under [runs/transition_rich_smoke](/home/mila/l/lia/skae/runs/transition_rich_smoke), so the new environments are now verified through the actual training CLI as well as through direct calibration.
- The standardized checkpoint evaluation path now also contains a deterministic flow-branching diagnostic for the frozen `2D` systems, with normalized tolerance sweeps, ground-truth-null thresholds, region-wise breakdowns, and companion `2D` visual artifacts.
- A new oracle state-space benchmark-validity read is now complete under
  [results/transition_rich_oracle_chart_switch_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_chart_switch_20260401).
  Its key consequence is a stricter benchmark ranking:
  - `gated_local_linear` is the clean main-text mechanistic positive
  - `gated_transfer_linear` should be written as the harder explicit-transfer
    stress test, not as the sole chart-switching flagship
  - `multiwell_strong_transition` remains a secondary toy
- A second oracle benchmark-validity read is now also complete under
  [results/transition_rich_oracle_refresh_cadence_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_refresh_cadence_20260401).
  It sharpens the same role split:
  - `gated_local_linear` is the clean periodic-refresh positive
  - `multiwell_strong_transition` is weaker and needs faster refresh
  - `gated_transfer_linear` is strongly cadence-sensitive and should be read as
    a hard transfer stress test whose relevant comparison is stale local charts
- The full seed-robust screening matrix on the calibrated three-system suite is
  now complete:
  - Stage 1 array `9135303` finished with `26/27` completed cells; the single
    failed task was `lista_dense_promoted_stage4 x multiwell_strong_transition x seed_2`
  - collect `9135304` completed `0:0`
  - comparisons `9135305-9135307` completed `0:0`
  - post hoc chart-change attribution `9135358` completed `0:0`
  - post hoc support local-linearity `9135411` completed `0:0`
- The collected read fixes the current paper-facing role split more sharply than
  the oracle-only stage:
  - `gated_local_linear` is the clean learned-model mechanistic positive
  - `gated_transfer_linear` is also a clean forecasting positive, but the
    stronger localization claim on true chart-change windows is negative
  - `multiwell_strong_transition` is not a clean forecasting positive at the
    paper `200k` budget
- The support local-linearity pass is encouraging supporting evidence for the
  branch, but it should be read as prior mechanism context rather than as the
  final fixed-`17` LISTA-versus-MLP reduction.
- The remaining clearly undercovered older mechanism artifacts are corrected competitive-LV support alignment under [results/zero_sparse_mechanisms_20260321/competitive_lv_representation_followup/support_alignment](/home/mila/l/lia/skae/results/zero_sparse_mechanisms_20260321/competitive_lv_representation_followup/support_alignment) with only seeds `0,1,2`, and the direct Kuramoto mode-support audit with only `5` seeds per root and sampling strategy; neither is the lead paper blocker anymore.
- Do not queue another broad benchmark or hard-system rerun by default. The
  immediate work is to turn the collected transition-rich read into a clean
  paper claim and plan any next runs only on the fixed `17`-system shortlist
  above. Do not reopen broader conceptual-inventory or full Claude-catalog
  selection for this branch; the only remaining selection question is the run
  order within the fixed shortlist and whether the single missing dense-LISTA
  `multiwell` seed is worth finishing.


## Consolidated Paper-Facing Families

For the NeurIPS draft's experiments section, prefer the evidence order in
[PAPER_EXPERIMENT_EVIDENCE_MAP.md](/home/mila/l/lia/skae/docs/PAPER_EXPERIMENT_EVIDENCE_MAP.md):
basin-support alignment, support-routed local prediction, long-horizon
forecasting competitiveness, then supporting/falsification diagnostics. The
family map below remains useful for provenance and artifact ownership, but it
should not dictate the paper's result order.

For drafting, compress the completed evidence into the family map below rather than citing each queue-era branch as its own experiment, and make the new lead branch explicit.

| Family | Merge these experiments | Main paper question | Writing rule |
|---|---|---|---|
| **Transition-rich basin partitioning** | tests-first toy-system calibration; fixed `17`-system shortlist; dense / block-diagonal LISTA live sweep; matched standard-MLP control; transition diagnostics; deterministic flow-consistency / flow-branching read; recurring-support local-linearity reuse metrics | Do LISTA encoders define reusable basin-aligned sparse supports on the fixed `17` systems, and is that basin separation stronger than for a matched standard MLP encoder? | This is the lead live family. Write it around basin separation first: count the systems where LISTA support views are basin-pure and reusable, use `gated_local_linear` as the clean positive anchor, treat `gated_transfer_linear` and the Claude subset as stress tests, and report the standard MLP control as the main contrast. |
| **Cross-system forecasting** | fair `200k` benchmark checkpoint family; matched zero-sparsity MLP benchmark extension; full-benchmark block-diagonal dense-opt transfer | What is the benchmark headline versus the MLP anchor once the dense comparator is fixed, and does explicit sparsity help beyond the same MLP with `lambda_sparse=0`? | Cite the fair `200k` benchmark as the supporting quantitative anchor. Do not let it crowd out the new transition-rich branch. |
| **Hard-system forecasting** | `dt`-rescue audit; focused smaller-`dt` Kuramoto/Hopfield follow-ups; long-horizon reevaluation of those same checkpoints; Kuramoto robustness/dimension sweeps; matched hard-system parity; matched block-diagonal fairness controls; higher-basin Hopfield / CLV robustness probes; matched zero-sparsity MLP hard-system extension | Where do step size, structure, and the sparsity penalty help, and where do LISTA families still fail? | Write this as one connected limitation/support family, not as a separate live execution branch. |
| **Appendix-only provenance** | dense LISTA recipe-selection/tuning sweeps; matched `50k` `v4` four-model audit | What tuning/provenance material justifies the fixed comparator choices and historical benchmark context? | Keep appendix-only. Do not present hyperparameter selection as a main result family. |

## Current Best Evidence

Paper-priority note:
- The experiments section should now be drafted from
  [PAPER_EXPERIMENT_EVIDENCE_MAP.md](/home/mila/l/lia/skae/docs/PAPER_EXPERIMENT_EVIDENCE_MAP.md)
  rather than from queue chronology: basin-support alignment first,
  support-routed prediction second, Dysts long-horizon forecasting third, and
  supporting/falsification diagnostics last.
- The live open branch is now the deterministic transition-rich basin-partitioning plan in [docs/planning/transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md).
- The evidence below remains the fixed supporting packet while that branch is
  being interpreted and written into the paper story.
- Main-text benchmark claims should be drawn from the `200k` results below.
- The older `50k` `v4` matrix is appendix-only historical context unless a reviewer specifically asks for the matched four-model snapshot.
- For the dense MLP-versus-LISTA benchmark, “fair” means matched systems, pass-2 `dt`, `200k` budget, `target_size=256`, `sequence_length=8`, and matched `[64,64]` front-end hidden widths/depths. It is not a strict equal-parameter comparison, because LISTA also keeps its learned recurrent encoder matrix on top of that shared front end.
- A matched zero-sparsity MLP extension of this same benchmark family is now collected under [results/paper_zero_sparse_benchmark_200k_20260321](/home/mila/l/lia/skae/results/paper_zero_sparse_benchmark_200k_20260321); it uses the same `200k` recipe as `generic_sparse_ns200k_best`, except `lambda_sparse = 0.0`, and the current table-facing forecasting packet now uses exact 10-seed coverage for every displayed benchmark and hard-system row. Its current read is mixed: zero-sparsity broadens good-system coverage on some late-horizon slices, but it is no longer the benchmark-median winner at `H1000`.

### Lead live branch: deterministic transition-rich basin partitioning

- The locked execution plan is [docs/planning/transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md).
- The current design-source companion for the unrun ablation axes is
  [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md).
  Treat it as planning ground truth for the next loop over items `3` and `4`,
  not as already-validated evidence.
- Training-budget policy for this live branch:
  use `20000` steps for forward interpretability diagnosis while the metrics
  and recipe are still being set; reserve `200000` only for the final locked
  paper-facing rerun.
- The branch should be written around basin separation on the fixed `17`
  systems, not around best-periodic forecasting.
- It should also not be written as a chart-switch-localization or sparse-only
  mechanism branch. Those are supporting side questions, not the branch
  acceptance criterion.
- The already-running April 7 queue is LISTA-only evidence. The matched
  standard MLP encoder control is still required before the intended fixed-`17`
  branch claim is decision-grade.
- The overnight continuation strengthens the same point: default `dt` is
  already adequate on the one-seed LISTA shortlist, and the repeated overnight
  `20k` reruns should be read as additional default-`dt` robustness checks
  rather than as smaller-`dt` rescue evidence.
- A paper-facing local reduction of the finished LISTA shortlist now exists
  under
  [basin_support_metrics_20260408_v3](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/basin_support_metrics_20260408_v3).
  The canonical prose definition note for these metrics is
  [docs/transition_rich_basin_support_metric_definitions.md](/home/mila/l/lia/skae/docs/transition_rich_basin_support_metric_definitions.md).
  Its branch-level read is clear:
  - mean support-group purity is `0.9883` for block-diagonal LISTA and
    `0.9961` for dense LISTA
  - mean retained-trajectory coverage is `0.8729` and `0.8787`, with
    `15/17` systems above the `0.60` coverage gate for both roots
  - local `H=20` NRMSE beats the matched global fit on `0/17` systems for
    both roots
  - local `H=20` NRMSE beats the shuffled baseline on `0/17`
    block-diagonal systems and `1/17` dense systems, and that lone dense win
    is `claude:duffing_triple_well` at only `0.1172` coverage
  - the current LISTA packet therefore supports a basin-pure recurring-support
    claim much more strongly than a local-linearity mechanism claim
- The first live LISTA basin-partition sweep on the fixed `17`-system shortlist
  was launched at each system's default `dt`, with a dependency-chained
  `dt`-halving rescue continuation if any arm fails `H1000 best-periodic < 50`.
  That already-launched April 7 sweep predates the new `20k` diagnostic-budget
  policy and should not be treated as the forward default.
- The older manual native-trio audit remains useful supporting detail, but it
  should now be subordinated to the full fixed-`17` reduction above:
  - all six dense/block-diagonal native arms pass the default-`dt` rescue gate
  - support groups remain basin-pure in every inspected native LISTA arm
  - `gated_local_linear` and `gated_transfer_linear` stay high-coverage purity
    positives
  - `multiwell_strong_transition` remains the weakest native case because
    coverage is still below the `0.60` gate
- The required test suite now exists and passes, along with the supporting calibration module [skae/transition_calibration.py](/home/mila/l/lia/skae/skae/transition_calibration.py) and the reproducible calibration entry point [tools/calibrate_transition_system.py](/home/mila/l/lia/skae/tools/calibrate_transition_system.py).
- Two native-plot `2D` deterministic candidates are now frozen from ground-truth calibration on the fixed `17x17` grid:
  - `multiwell_strong_transition`
    - `5` endpoint basins
    - per-endpoint crossing fractions `0.500 / 0.500 / 0.500 / 0.500 / 0.585`
    - overall crossing `0.512`
    - label stability `1.000`
  - `gated_local_linear`
    - `3` endpoint basins
    - per-endpoint crossing fractions `0.670 / 0.577 / 0.539`
    - overall crossing `0.599`
    - label stability `1.000`
- The explicit-transfer toy is now implemented and calibrated:
  - `gated_transfer_linear`
    - `3` endpoint basins
    - source-neighborhood transfer fractions `0.333 / 0.444 / 0.333`
    - overall source-neighborhood transfer `0.370`
    - core-retention fractions `1.000 / 1.000 / 1.000`
    - overall core retention `1.000`
    - label stability `1.000`
- The paper-usable mechanics figures for that transfer toy now exist:
  - [gated_transfer_linear_region_map.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_region_map.svg)
  - [gated_transfer_linear_chart_trajectories.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_chart_trajectories.svg)
  - [gated_transfer_linear_endpoint_trajectories.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_endpoint_trajectories.svg)
  - [gated_transfer_linear_transfer_summary.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_transfer_summary.svg)
- The same calibration gives one current caveat:
  - `multiwell_gradient` under the new corridor-aware labeling is not yet a clean low-transition control (`overall crossing = 0.401`, center-basin crossing `0.242`), so it should remain a secondary reference until we decide whether to retune it.
- The branch now also has a paper-usable mechanics figure:
  - [transition_rich_100_trajectories.svg](/home/mila/l/lia/skae/docs/figures/transition_rich_20260331/transition_rich_100_trajectories.svg)
- Updated interpretation of those two frozen systems:
  - they are good secondary toys for transition handling and local chart
    mechanics
- A new oracle chart-switch benchmark-validity read is now complete under
  [results/transition_rich_oracle_chart_switch_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_chart_switch_20260401):
  - `gated_local_linear`
    - oracle-vs-global gain at `H20` on all starts: `0.129974`
    - oracle-vs-fixed-chart gain at `H20` on all starts: `0.0219174`
  - `gated_transfer_linear`
    - oracle-vs-global gain at `H20` on all starts: `0.0206368`
    - oracle-vs-fixed-chart gain at `H20` on all starts: `0.00237432`
    - oracle-vs-global gain on transfer starts: `-0.0557357`
    - oracle-vs-fixed-chart gain on transfer starts: `0.00993001`
  - `multiwell_strong_transition`
    - oracle-vs-global gain at `H20` on all starts: `0.0144757`
    - oracle-vs-fixed-chart gain at `H20` on all starts: `0.00539429`
- Revised benchmark role split:
  - `gated_local_linear` is the cleanest chart-switching positive
  - `gated_transfer_linear` is the explicit-transfer stress test
  - `multiwell_strong_transition` is the weaker shared-corridor toy
- A second oracle benchmark-validity read is now complete under
  [results/transition_rich_oracle_refresh_cadence_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_refresh_cadence_20260401):
  - `gated_local_linear`
    - median dwell: `171`
    - recovered oracle fraction at `H20`: `0.988651` at `c=2`, `0.90577` at
      `c=5`, `0.665843` at `c=10`
    - largest cadence preserving at least `90%` of the oracle-vs-fixed gain on
      all starts: `c=5`
  - `multiwell_strong_transition`
    - median dwell: `92`
    - recovered oracle fraction at `H20`: `0.968223` at `c=2`, `0.858159` at
      `c=5`, `0.639087` at `c=10`
    - largest cadence preserving at least `90%` of the oracle-vs-fixed gain on
      all starts: `c=2`
  - `gated_transfer_linear`
    - median dwell: `33`
    - recovered oracle fraction at `H20`: `0.88425` at `c=2`, `0.503731` at
      `c=5`, `0.0577701` at `c=10`
    - largest cadence preserving at least `90%` of the oracle-vs-fixed gain on
      all starts: only `c=1`
    - on transfer starts, oracle refresh still beats stale local charts but not
      the global fit
- The standardized evaluation stack now has one additional forecast-side diagnostic for these systems:
  - sweep a normalized same-state tolerance on pooled rollout states
  - set the next-step divergence threshold from the ground-truth simulator so the true system is exactly zero under the chosen sweep
  - report close-pair counts, flow-branching rate, next-step / `K`-step divergence severity, and basin-core vs transition-region breakdowns
- First model pass is also fixed:
  - `generic_sparse`
  - matched zero-sparsity MLP
  - promoted dense LISTA
  - `lista_blockdiag` only on the strongest positive case if the screening read is genuinely informative
- The Stage 1 execution wrapper and branch-specific summary collector now exist:
  - [tools/build_transition_rich_screening_tasks.py](/home/mila/l/lia/skae/tools/build_transition_rich_screening_tasks.py)
  - [tools/summarize_transition_rich_screening.py](/home/mila/l/lia/skae/tools/summarize_transition_rich_screening.py)
  - [scripts/collect_transition_rich_screening.sh](/home/mila/l/lia/skae/scripts/collect_transition_rich_screening.sh)
  - [scripts/queue_transition_rich_screening_stage1_20260401.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_screening_stage1_20260401.sh)
- The post hoc chart-change attribution pass now also exists as a separate,
  scientifically cleaner stage:
  - [tools/evaluate_transition_rich_chart_change_attribution.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_chart_change_attribution.py)
  - [tools/collect_transition_rich_chart_change_attribution.py](/home/mila/l/lia/skae/tools/collect_transition_rich_chart_change_attribution.py)
  - [scripts/collect_transition_rich_chart_change_attribution.sh](/home/mila/l/lia/skae/scripts/collect_transition_rich_chart_change_attribution.sh)
  - [scripts/queue_transition_rich_chart_change_attribution_20260401.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_chart_change_attribution_20260401.sh)
- The post hoc support-local-linearity pass now also exists as a separate,
  scientifically cleaner stage:
  - [tools/evaluate_transition_rich_support_local_linearity.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_support_local_linearity.py)
  - [tools/collect_transition_rich_support_local_linearity.py](/home/mila/l/lia/skae/tools/collect_transition_rich_support_local_linearity.py)
  - [scripts/collect_transition_rich_support_local_linearity.sh](/home/mila/l/lia/skae/scripts/collect_transition_rich_support_local_linearity.sh)
  - [scripts/queue_transition_rich_support_local_linearity_20260401.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_support_local_linearity_20260401.sh)
- Undertrained smoke validation exists only as a tooling check, not as a paper
  result, under
  [results/transition_rich_smoke_chart_change_attr_20260401](/home/mila/l/lia/skae/results/transition_rich_smoke_chart_change_attr_20260401):
  it shows a small positive chart-change localization delta but no meaningful
  transfer-trajectory gain, which is exactly the expected “pipeline works,
  science not claimed yet” outcome for a `5`-step smoke root.
- Undertrained support-local-linearity smoke validation also exists only as a
  tooling check, not as a paper result, under
  [results/transition_rich_smoke_support_local_linearity_20260401](/home/mila/l/lia/skae/results/transition_rich_smoke_support_local_linearity_20260401):
  on the collected `generic_sparse` transfer smoke, retained support groups
  cover `0.858` of all trajectories, `0.865` of chart-switch trajectories, and
  `0.700` of transfer trajectories, with `H20` local/global/shuffled latent
  NRMSE `0.0701 / 0.1309 / 0.2018` and weighted endpoint / source-endpoint-pair
  purity `1.000 / 0.891`; that is the right nondegenerate tooling read, but it
  is still not a paper claim because the smoke checkpoints trained for only `5`
  steps.
- The collected Stage 1 screen is now the real model-side read:
  - `26/27` runs completed under
    [results/transition_rich_screening_stage1_20260401](/home/mila/l/lia/skae/results/transition_rich_screening_stage1_20260401)
  - `gated_local_linear` is a clean positive for all three roots, with
    `H1000` best-periodic medians `0.0890151`, `0.000710212`, `0.000783794`
    against no-reencode medians `27651.5`, `2.86748`, `206.989`
  - `gated_transfer_linear` is also a clean positive for all three roots, with
    `H1000` best-periodic medians `1.7954`, `1.80077`, `1.09863` against
    no-reencode medians `6.54401e+20`, `2.45499e+34`, `1.25639e+12`
  - `multiwell_strong_transition` is not a clean forecasting positive at the
    paper budget; `H100` and `H500` best-periodic medians remain unstable and
    enormous across roots
- The collected chart-change attribution pass is now also the real model-side
  causal read on the explicit-transfer toy:
  - summary:
    [results/transition_rich_chart_change_attribution_20260401/summary.md](/home/mila/l/lia/skae/results/transition_rich_chart_change_attribution_20260401/summary.md)
  - all three localization deltas are negative for all three roots, so the
    stronger claim that learned periodic gains localize at true chart-change
    windows is not currently supported
- The collected support local-linearity pass is also now the real
  partition-reuse read:
  - summary:
    [results/transition_rich_support_local_linearity_20260401/summary.md](/home/mila/l/lia/skae/results/transition_rich_support_local_linearity_20260401/summary.md)
  - `gated_local_linear` is the strongest reusable-partitions positive,
    `gated_transfer_linear` is intermediate, and `multiwell_strong_transition`
    is weakest on retained-coverage
  - the matched zero-sparsity MLP is also strong, so this is not a sparse-only
    mechanism win
- Interpretation:
  - the branch now has a valid three-system deterministic suite before any
    model training
  - the suite now has two independent ground-truth benchmark-validity reads:
    one for per-step chart switching and one for periodic refresh cadence
  - the branch now also has the first collected model-side diagnostic read, and
    it supports a narrower claim than the most ambitious version of the story:
    strong periodic-refresh positives on `gated_local_linear` and
    `gated_transfer_linear`, but not a clean `multiwell_strong_transition`
    positive and not a chart-localization win on the transfer toy
- The supporting benchmark, hard-system, and mechanism evidence below remains the context for interpreting this new branch.

### 1. Supporting benchmark packet: one fair `200k` checkpoint family with `H100-H3000` coverage

- The table-facing benchmark family should now be read from [results/paper_zero_sparse_benchmark_200k_20260321/collect/forecasting_rows.csv](/home/mila/l/lia/skae/results/paper_zero_sparse_benchmark_200k_20260321/collect/forecasting_rows.csv), [results/paper_zero_sparse_benchmark_200k_20260321/collect/paper_benchmark_summary.json](/home/mila/l/lia/skae/results/paper_zero_sparse_benchmark_200k_20260321/collect/paper_benchmark_summary.json), and [results/paper_zero_sparse_benchmark_200k_20260321/fixed_cadence_periodic_100/fixed_cadence_summary.json](/home/mila/l/lia/skae/results/paper_zero_sparse_benchmark_200k_20260321/fixed_cadence_periodic_100/fixed_cadence_summary.json), not from the older mixed-coverage benchmark summaries.
- The verified best-periodic cross-system medians are:
  - sparse MLP: `2.947e-4 / 0.0051 / 0.0240 / 0.0795 / 0.1123 / 0.1591 / 0.2201`
  - zero-sparsity MLP: `3.021e-4 / 0.0063 / 0.0353 / 0.1067 / 0.1351 / 0.1738 / 0.1951`
  - promoted dense LISTA: `3.440e-4 / 0.0047 / 0.0250 / 0.0449 / 0.0627 / 0.0880 / 0.1039`
  - horizons are `H100/H500/H1000/H1500/H2000/H2500/H3000`
- The verified good-system counts are:
  - sparse MLP: `27/25/25/26/25/25/25`
  - zero-sparsity MLP: `27/26/26/27/27/27/27`
  - promoted dense LISTA: `28/25/26/25/25/25/25`
- Interpretation:
  - sparse MLP is still the clean anchor at `H100` and `H1000`
  - promoted dense LISTA is the strongest cross-system LISTA result and is median-best at `H500` and `H1500-H3000`
  - zero-sparsity does not produce the benchmark-median `H1000` win implied by the earlier mixed-coverage snapshot, but it does improve late-horizon coverage on more systems than it wins by median
  - the paper therefore needs a genuine three-way benchmark read rather than a sparse-versus-dense summary plus a side-note control
- Under one global deployment-like cadence `periodic_100`, the verified late-horizon medians are:
  - sparse MLP: `0.1070 / 0.2140 / 0.3690 / 0.6148`
  - zero-sparsity MLP: `0.1729 / 0.4542 / 0.7041 / 0.9349`
  - promoted dense LISTA: `0.0744 / 0.2688 / 0.4437 / 0.6508`
  - horizons are `H1500/H2000/H2500/H3000`
- Fixed-cadence interpretation:
  - explicit sparsity still helps materially relative to the matched zero-sparsity MLP under one deployment-like cadence
  - promoted dense LISTA remains the late-horizon fixed-cadence median winner in the benchmark packet, so the deployment-style story is again three-way rather than a sparse-only positive
- The dense-optimizer block-diagonal transfer still does not rescue block-diagonal LISTA globally; keep block-diagonal claims restricted to targeted hard-system evidence and explicit fairness-control tables.

### 2. Appendix-only dense LISTA comparator-selection provenance

This is supporting provenance for the dense comparator used in Section 1. It should be mentioned only briefly in the main text and detailed in the appendix if needed.

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
  - for paper writing, the fair `200k` comparison in Section 1 is the dense-vs-MLP headline; the older Stage-4-vs-`v4` comparison is supporting evidence for recipe promotion, not the main-text benchmark claim
  - when writing that headline, say explicitly that the dense-vs-MLP comparison is width/depth-aligned but not parameter-matched
  - the remaining dense-LISTA paper risk is no longer “can it catch up on the easier systems?”; it is whether the paper cleanly separates the cross-system parity win from the unresolved hard-system failures

### 3. `dt` rescue and hard-system follow-up are now decision-grade: step size is a real bottleneck, Kuramoto has a targeted `200k` rescue, and Hopfield remains MLP-better

- A matched zero-sparsity MLP extension of every paper-facing hard-system setting cited in this section is now collected under [results/zero_sparse_hard_systems_20260321](/home/mila/l/lia/skae/results/zero_sparse_hard_systems_20260321). It covers Kuramoto (`N=16` identical, `N=16` uniform-spread, and the dimension sweep), Hopfield (`N=16` and quarter-step `N=64`), corrected 4-basin competitive LV, and fixed 8-basin competitive LV at both cited step sizes. The recovery chain cleared on March 22 when requeued task `9007966_4` completed and refreshed collectors/comparisons `9016645-9016660` all finished `0:0`.

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
  - step size is a real scientific bottleneck, not just a scheduling artifact
  - the open paper problem shifted from "finish the rerun" to "what to do with systems that stay hard even after the allowed `dt` rescue"

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
- Completed focused Kuramoto `dt=0.00625`, `200k` follow-up under the official short-horizon compare [Kuramoto comparison](/home/mila/l/lia/skae/results/kuramoto_dt00625_200k_compare_20260308/compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md) plus the later long-horizon collector [Kuramoto forecasting summary](/home/mila/l/lia/skae/results/kuramoto_dt00625_200k_compare_20260308/collect/forecasting_summary.md):
  - `generic_sparse`: seed-median `H1000` best-periodic `27.02`
  - dense LISTA: seed-median `H1000` best-periodic `13.84`
  - `lista_blockdiag`: seed-median `H1000` best-periodic `6.98`
  - all five `lista_blockdiag` seeds are good and tightly clustered in `6.89-7.13`
  - the underlying focused summary is again not `H1000`-only: at `H100`, `generic_sparse=0.0343`, `lista_blockdiag=0.1536`, `lista_dense=0.2194`; at `H500`, `generic_sparse=1.4972`, `lista_blockdiag=2.6834`, `lista_dense=4.1222`; the `lista_blockdiag` win appears only at `H1000`
  - the same run family now also has explicit `H1500/H2000/H2500/H3000` medians from the collector: `generic_sparse = 547.37 / 1.208e+04 / 3.370e+05 / 9.207e+06`, `lista_blockdiag = 10.93 / 14.52 / 17.94 / 21.58`, dense LISTA `= 54.85 / 205.26 / 541.19 / 1519.09`
  - **Fixed-cadence ablation (completed):** `periodic_100` exactly reproduces the official `best_periodic` `H1000` ranking for all three roots (`6.98`, `13.84`, `27.02`), so the long-horizon Kuramoto block-diagonal win is already a fixed-cadence result, not a `best_periodic` oracle artifact. At `H500`, `generic_sparse` still has lower error than `lista_blockdiag` under the fixed cadence.
  - **Checkpoint-selection ablation (completed):** switching from `evaluation_results_best.json` to `evaluation_results_last.json` on the current `dt=0.00625`, `200k` comparison does not change the model ranking or good-band membership (`lista_blockdiag`: `6.98 -> 7.00`, `lista_dense`: `13.84 -> 17.63`, `generic_sparse`: `27.02 -> 29.43`). The older `dt=0.0125` pilot mismatch (`23.40 -> 14.64`) is real but tied to the earlier, superseded setting. Keep `evaluation_results_best.json` as the official paper rule.
- Completed Hopfield `dt=0.00625`, `200k` follow-up under [Hopfield follow-up summary](/home/mila/l/lia/skae/results/hopfield_dt00625_200k_compare_20260309/forecasting_summary.md):
  - `generic_sparse`: seed-median `H1000` best-periodic `3.36`
  - `lista_blockdiag`: seed-median `H1000` best-periodic `8.82`
  - the same file shows the MLP lead at every collected horizon: `H100` `0.0500` vs `0.1075`, `H500` `0.8835` vs `3.5303`, `H1000` `3.3642` vs `8.8212`
  - the long-horizon continuation stays negative as well: `generic_sparse` is `6.61 / 9.17 / 10.96 / 12.23` at `H1500/H2000/H2500/H3000`, versus `12.23 / 12.83 / 13.23 / 13.58` for `lista_blockdiag`
  - both are inside the good-forecast band on the system median, but every-step errors are still enormous for both
- The completed `H3000` reevaluation sharpens the hard-system long-horizon read across the whole Kuramoto/Hopfield family:
  - Kuramoto `N=16`, `dt=0.00625`, `200k`: `lista_blockdiag` remains much better than `generic_sparse` across `H1500/H2000/H2500/H3000` (`10.93 / 14.52 / 17.94 / 21.58` vs `547.37 / 1.208e+04 / 3.370e+05 / 9.207e+06`), but it is out of band beyond `H1000`
  - Kuramoto `N=32`, `dt=0.00625`, `200k`: `lista_blockdiag` stays better across `H1500/H2000/H2500/H3000` (`10.89 / 16.14 / 21.69 / 27.95` vs `13.60 / 24.27 / 42.19 / 75.25` for `generic_sparse`), but it is also out of band at those later horizons
  - Kuramoto uniform-spread `N=16`: the repaired `generic_sparse_blockdiag` mirror is now slightly better than `lista_blockdiag` at every reported horizon (`8.13 / 24.76 / 91.23 / 399.74 / 1724.45` versus `9.53 / 28.41 / 117.46 / 523.72 / 2129.37`), but both repaired structured roots are still catastrophic by `H3000`
  - Hopfield `N=16`, `dt=0.00625`, `200k`: `generic_sparse` stays better at `H1500/H2000/H2500/H3000` (`6.61 / 9.17 / 10.96 / 12.23`) than `lista_blockdiag` (`12.23 / 12.83 / 13.23 / 13.58`); only `generic_sparse` stays in-band through `H2000`
  - higher-basin Hopfield `N=64`, `dt=0.0015625`, `200k`: quarter-`dt` improves errors but remains strongly negative for LISTA, with `generic_sparse = 309.92 / 520.71 / 711.04 / 873.50` at `H1500/H2000/H2500/H3000` versus dense `591.50 / 842.59 / 1046.18 / 1206.93` and targeted blockdiag `735.26 / 1035.51 / 1274.93 / 1461.51`
- Completed Kuramoto `N=32`, `dt=0.00625`, `200k`, `3`-seed confirmation under the short-horizon compare [Kuramoto `N=32` summary](/home/mila/l/lia/skae/results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md) and long-horizon collector [Kuramoto `N=32` forecasting summary](/home/mila/l/lia/skae/results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/collect/forecasting_summary.md):
  - `generic_sparse`: seed-median `H1000` best-periodic `6.65` (all seeds good)
  - `lista_blockdiag`: seed-median `H1000` best-periodic `6.00` (all seeds good, std `0.33`)
- Completed Kuramoto dimension sweep under [Kuramoto dimension summary](/home/mila/l/lia/skae/results/kuramoto_dimension_sweep_dt00625_200k_20260309/collect/kuramoto_dimension_summary.md):
  - dimensions: `N={8,16,24,32,64}`
  - models: `generic_sparse`, promoted dense LISTA, `lista_blockdiag`
  - fixed setting: `dt=0.00625`, `200k`, `5` seeds
  - `H1000` seed-median best-periodic by dimension:
    - `generic_sparse`: `813.57`, `30.18`, `6.71`, `6.68`, `208.93`
    - promoted dense LISTA: `495.07`, `13.44`, `14.99`, `92.28`, `208.71`
    - `lista_blockdiag`: `8.11`, `7.07`, `6.57`, `5.92`, `23.27`
  - seed robustness:
    - `lista_blockdiag` is all-seeds-good at `N=16/24/32`
    - `lista_blockdiag` is median-good but not fully robust at `N=8` (`4/5` good seeds, worst seed `10.89`)
    - `lista_blockdiag` falls out of band at `N=64` (`2/5` good seeds, worst seed `209.20`)
- Completed repaired block-diagonal MLP mirrors for the missing Kuramoto families under the retry1 roots:
  - uniform-spread `N=16`: `generic_sparse_blockdiag` reaches `H1000 = 8.13` with all `5/5` seeds good, beating both `generic_sparse` (`44.46`) and `lista_blockdiag` (`9.53`)
  - dimension sweep `H1000`: `generic_sparse_blockdiag = 10.61 / 6.51 / 5.79 / 5.16 / 208.54` at `N=8/16/24/32/64`
  - on that sweep, the repaired MLP block-`K` mirror beats `generic_sparse` at every `N`, beats `lista_blockdiag` at `N=16/24/32`, but not at `N=8` or `N=64`
  - the suspicious `N=64` row is now audited and decision-grade: it is a real seed-fragility limit (`2/5` good seeds, `3/5` collapsed seeds), not a leftover March 17 invalid-control artifact
- Interpretation:
  - smaller `dt` is the dominant hard-system lever in the current regime
  - `lista_blockdiag` is the strongest hard-system LISTA result on Kuramoto and the only model that cleanly wins the `N=16` three-way comparison there at `H1000`; the shorter-horizon audit shows this is a specifically long-horizon advantage rather than across-horizon dominance
  - the completed repaired fairness mirrors narrow the causal read: block structure alone helps on the targeted `N=16` family, on uniform-spread `N=16`, and across the sweep at `N=16/24/32`, but it does not rescue `N=8` and the `N=64` failure is real but seed-sensitive rather than a provenance glitch
  - promoted dense LISTA does not transfer as a robust Kuramoto solution under this sweep
  - Hopfield is no longer a catastrophic boundary case in the targeted `N=16`, `dt=0.00625`, `200k` setting, but it is still not a structured-LISTA success story because `generic_sparse` remains better through `H2000` and both models are out of band by `H2500-H3000`
  - the remaining scientific limitation is autonomous rollout stability, not whether periodic reencoding can rescue the hard systems at all
  - the remaining Kuramoto paper question is no longer missing fairness controls; it is how to present a targeted moderate-dimension success with an explicit `N=8` and `N=64` limit
- Completed Kuramoto robustness evaluation (uniform frequency spread, `N=16`, `dt=0.00625`, `200k`, `5` seeds) under the short-horizon compare [Kuramoto robustness comparison](/home/mila/l/lia/skae/results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/compare/lista_blockdiag_uniform_spread_vs_generic_sparse_uniform_spread/forecasting_comparison.md) and long-horizon collector [Kuramoto robustness forecasting summary](/home/mila/l/lia/skae/results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/collect/forecasting_summary.md):
  - `lista_blockdiag`: seed-median `H1000` best-periodic `9.53` (4/5 seeds good, std `0.64`)
  - `generic_sparse`: seed-median `H1000` best-periodic `44.46` (0/5 seeds good)
  - `4.7x` improvement; the Kuramoto block-diagonal positive is not a single-regime artifact
  - every-step errors remain catastrophic for both models under heterogeneity; periodic re-encoding is essential

### 4. Appendix-only `50k` `v4` audit: useful for matched four-model symmetry, not for main-text claims

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
  - `v4` is still a useful matched four-model snapshot and the cleanest source for the `dt`-resolution audit.
  - It should be treated as appendix-only historical context once the `200k` reruns materially improve the same story.
  - Do not use `v4` as the source of main-text rankings or headline model comparisons when a `200k` counterpart exists.

## Recent Queue Activity

The March 9-20 paper-strengthening program is fully closed and now belongs to the archived planning set. The only live successor branch is the deterministic transition-rich basin-partitioning plan documented at the top of this file.

### Just Closed

- **Kuramoto uniform-spread dense-LISTA completion run:** the March 20 appendix-table audit found that the original March 9 uniform-spread robustness manifest never launched dense LISTA, so the missing row was genuine rather than a collector omission. Completion chain `8989946 -> 8989947 -> 8989948/8989949/8989950` finished cleanly under [results/kuramoto_uniform_spread_dense_20260320](/home/mila/l/lia/skae/results/kuramoto_uniform_spread_dense_20260320). Training tasks `8989946_[0-4]` ended between `2026-03-20 02:58 EDT` and `2026-03-20 03:31 EDT`; collector `8989947` finished at `03:32 EDT`, and compare jobs `8989948/8989949/8989950` finished at `03:33 EDT`. The completed row is `0.2081 / 4.1470 / 16.55 / 46.38 / 132.35 / 415.24 / 1530.83` at `H100/H500/H1000/H1500/H2000/H2500/H3000`.
- **Kuramoto uniform-spread valid MLP `+ block-K` rerun:** retry chain `8988801 -> 8988802 -> 8988803/8988804` closed successfully under [results/kuramoto_uniform_spread_gs_blockdiag_retry1_20260319](/home/mila/l/lia/skae/results/kuramoto_uniform_spread_gs_blockdiag_retry1_20260319). Training tasks ended between `2026-03-19 22:33 EDT` and `2026-03-19 23:47 EDT`; the collector finished at `23:48 EDT`, and both compare jobs finished at `23:49 EDT`.
- **Kuramoto dimension-sweep valid MLP `+ block-K` rerun:** retry chain `8988805 -> 8988806 -> 8988807/8988808` also closed successfully under [results/kuramoto_dimension_sweep_gs_blockdiag_retry1_20260319](/home/mila/l/lia/skae/results/kuramoto_dimension_sweep_gs_blockdiag_retry1_20260319). Training tasks ended between `2026-03-19 22:06 EDT` and `2026-03-19 23:54 EDT`; the collector finished at `23:54 EDT`, and both compare jobs finished at `23:55 EDT`.
- **Queue hygiene note:** the first submission attempt (`8988754`, `8988758`) was invalidated by external startup termination after accidental duplicate-submission cleanup. Use only the retry1 roots above for paper-facing interpretation.

### Last 48 hours (completed state)

- **Paper-facing horizon reevaluation:** closed successfully after one tooling repair. Original kickoff array `8929393` failed `300/300` tasks because `run_manifest.tsv` used CRLF line endings. The repaired chain `8933469 -> 8933470 -> 8935507 -> 8936342 -> 8937045 -> 8937956 -> 8938574` completed all `1528` reevaluations and refreshed the benchmark / compare artifacts through `H3000`.
- **Hard-system parity sweep:** closed successfully after a small anchor repair. Stage-2 confirm array `8931671` ended with two failed anchor rows (`87`, `137`); the repair / recollect / resummary chain `8933794 -> 8933795 -> 8933796` completed and produced the final negative-for-LISTA parity summary now used in this document.
- **Corrected `competitive_lv` representation follow-up:** closed after one repair. Wrapper `8933876` completed, support-alignment job `8939086` completed, the first label-free clustering array `8939087` failed `39/39` because checkpoint paths in `lfc_task_specs.tsv` retained a trailing carriage return, and March 13 repair chain `8945129 -> 8945133 -> 8945135` then completed cleanly. The final scientific read is still negative on the clean `4`-basin paper roots.
- **Hopfield quarter-`dt` higher-basin rerun:** fully complete. Base sweep `8926091` and matched fairness control `8926089` both finished; the smaller `dt` improves errors materially but does not move any root into the good band or change the ordering (`generic_sparse` remains best).
- **Other completed paper-side jobs:** the historical matched block-diagonal fairness-control jobs (`8922796`, `8922810`, `8922811`) all finished at the queue level, but their MLP `+ block-K` outputs were later invalidated by the March 17 audit; the CLV high-basin scalability / smaller-`dt` sweeps (`8923108`, `8926090`) and the clean 4-basin CLV retrain / recovery / extension chain (`8922033`, `8922374`, `8922879`) are complete and already reflected in the current paper claims above.
- **Accounting note:** the only other failed job in the last-48-hour accounting window was one unrelated `interactive` shell session (`8936456`); it does not affect paper state.

### Recently completed

- **Paper-facing `H3000` horizon reevaluation:** complete. The refreshed fair benchmark is now split by horizon: `generic_sparse_ns200k_best` is best by cross-system median at `H100/H500/H1000` (`3.146e-4`, `0.0050`, `0.0233`), while promoted dense Stage 4 becomes median-best at `H2000/H2500/H3000` (`0.0627`, `0.0910`, `0.0940`) but falls to `24/29` good systems at `H3000` versus `26/29` for the MLP anchor.
- **Long-horizon fixed-cadence benchmark rescore:** complete offline from the refreshed fair `200k` benchmark JSONs; no queue submission was needed. Under one global `periodic_100`, dense remains breadth-competitive at `H1500-H3000` but no longer has a clean late-horizon median edge: the MLP is better by cross-system median at `H2000/H2500`, and at `H3000` the MLP also keeps better coverage (`20/29` vs `17/29`).
- **Hard-system parity sweep:** complete. `generic_sparse` is best on all `14` confirmed settings; dense LISTA records `0` wins / `10` losses / `4` worse-seed-robustness rows, and `lista_blockdiag` records `0` wins / `8` losses / `6` worse-seed-robustness rows.
- **Corrected `competitive_lv` representation follow-up:** complete. The repaired March 13 clustering rerun wrote the missing `label_free_clustering_v2/` outputs, but the corrected CLV result remains negative for the intended support-based narrative: no clean `4`-basin paper root clears the support-view gate, all keep negative cosine separation, and support-view clustering collapses to `5/2/1` discovered groups across seeds.
- **Competitive LV high-basin smaller-`dt` halving rerun:** complete on array `8926090` (`96/96`). Halving to `dt=0.0025` removes the remaining fixed-system `200k` `lista_blockdiag` seed failures on the higher-basin CLV probes, but `generic_sparse` remains best and `generic_sparse_blockdiag` stays neutral.
- **Kuramoto unique mode-support audit (completed March 10):** All `30/30` array tasks and collector finished under `results/kuramoto_mode_support_audit_20260310/`. The strong negative claim is confirmed: Kuramoto winding-number basins do not have meaningful basin-specific support patterns. Mode-support uniqueness is trivially degenerate — every trajectory has its own unique support (`traj_unique=1.0`), mode supports are singletons, basin consistency is negligible (`0.0625` balanced / `0.309` random), and Hamming geometry is flat (`ratio≈1.0`). This holds identically across all 3 model families (`generic_sparse`, `lista_dense`, `lista_blockdiag`), all 5 seeds, both sampling protocols (`random`, `balanced`), all support modes (`mean`, `majority`, `modal`), and all threshold values (`1e-4` to `1e-1`). This closes the gap left by label-free clustering v2 and directly confirms both claims: basins are not recoverable from latent features, and basins do not have literal reusable mode supports.
- **Broad support-alignment audit on labelable `v4` systems:** complete under [support-alignment summary](/home/mila/l/lia/skae/results/paper_benchmark_support_alignment_20260311_v4_labelable/summary.md). Across `11` valid labelable systems (`132` checkpoints), binary `mode_uniqueness_rate` saturated at `1.0` on all `44/44` system-root medians, while cosine separation still cleanly split the systems: multiwell positive, Duffing negative, Kuramoto negative, and Hopfield mixed.
- **Fair `200k` follow-up benchmark:** complete. `generic_sparse_ns200k_best` is the best full-benchmark root by cross-system median `H1000` best-periodic (`0.0233`), while promoted dense Stage 4 still wins `18/29` shared systems and keeps `26/29` good systems at that horizon.
- **Dense LISTA promoted Stage 4 rerun:** complete. One fixed fair dense recipe wins `21/29` systems against the fixed `generic_sparse` anchor and improves the dense median to `0.0232`.
- **Kuramoto dimension sweep:** complete. `lista_blockdiag` is robust through `N=32`, not fully robust at `N=8`, and no longer in-band at `N=64`.
- **Focused Kuramoto and Hopfield smaller-`dt` follow-ups:** complete. `lista_blockdiag` wins the Kuramoto `dt=0.00625`, `200k` comparison at `H1000`, but the refreshed `H1500-H3000` read reduces that win to a relative-only advantage; Hopfield remains `generic_sparse`-best through `H2000` and both models fail by `H2500-H3000`.
- **Label-free clustering v2:** complete on array `8919951` with collector `8919952`. Multiwell systems are strongly positive, Duffing is weakly positive, and Kuramoto is negative.

### 6. Broad support-alignment audit: binary mode uniqueness saturates, cosine separation carries the real signal

Together with the corrected `competitive_lv` representation rerun and the March 13 local-linearity study, this section belongs to the basin-support/mechanism family rather than a standalone side analysis.

- A matched zero-sparsity MLP extension of this mechanism family is now complete under [results/zero_sparse_mechanisms_20260321](/home/mila/l/lia/skae/results/zero_sparse_mechanisms_20260321). The original wrapper `9007983` was canceled during March 22 recovery, but the refreshed wrapper `9016661` finished the same support-alignment, label-free clustering, Kuramoto support-audit, and recurring-support local-linearity analyses on the no-sparsity MLP checkpoints. The completed Kuramoto support-audit rerun shows that removing the sparsity penalty does not recover reusable mode supports.

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

This is the label-free clustering subfamily inside the same basin-support/mechanism family.

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
- **Competitive LV multi-basin retrain and representation family (forecasting complete March 11; representation family complete March 13):** The previous `competitive_lv` benchmark was trivial (1 observed basin at `INTERACTION_SCALE=0.35`). The config is now `0.70`, producing 4 major basins. All 28/28 training tasks completed (arrays `8922033` + recovery `8922374`). Forecasting collection and comparison are done. All models are inside the good-forecast band; the paper-facing `200k` result is `generic_sparse=0.1254`, with `lista_blockdiag` and `lista_dense` close behind. The older `50k` comparison is only an auxiliary sanity check. Competitive LV is not a problematic forecasting holdout for any architecture. The related representation family is now also complete: support alignment on the corrected checkpoints remains weak/trivial, the repaired March 13 label-free clustering v2 rerun finished cleanly, and the final clustering read stays negative for stable support-defined basin recovery on the clean `4`-basin paper roots. Do NOT re-use any old 1-basin `competitive_lv` evaluation results.

## Decision Rules

- If default `dt` is poor and the benchmark rescue chain requests halving, prefer **smaller `dt`** before broader model changes or `10x` longer training.
- **The `200k` results are now the primary paper evidence.** The `v4` `50k` matrix is retained only as appendix / historical context for the matched four-model audit. All headline paper claims, cross-system rankings, and model comparisons should be drawn from the `200k` runs in `results/paper_followup_recipes_200k_20260309`.
- If a `200k` rerun materially improves over a `50k` result, do not foreground the `50k` result in the paper body.
- Use `generic_sparse_ns200k_best` as the primary paper anchor (not the `50k` `generic_sparse`).
- Use the promoted dense Stage-4 root (`lista_dense_promoted_stage4`) as the primary dense LISTA comparator.
- Use dense LISTA as the cross-system LISTA reference, but keep `lista_blockdiag` as the only LISTA-family candidate for intrinsic-HD follow-up unless new evidence clearly overturns that ranking.
- Treat the completed dense-LISTA easy-system Stage-1 sweep as evidence that external optimization alone can recover most easy dense-LISTA near-misses without changing architecture or `dt`.
- Treat the completed dense-LISTA Stage-2 / Stage-4 chain as the parity decision point:
  - stop coefficient-only holdout tuning
  - promote `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3` as the single fair dense recipe
  - use the completed Stage-4 rerun as the current dense parity evidence (`21/29` wins, `26/29` good systems, `0` dense-fails-anchor-passes systems)
  - when discussing the fair `200k` comparison, make the split explicit: dense wins more systems and keeps more systems good, while `generic_sparse_ns200k_best` has the best overall median
  - `v4` remains an appendix-only symmetric four-model audit; the `200k` follow-up is the primary paper-facing comparison
- Treat the full-benchmark block-diagonal dense-opt transfer as a negative result for global parity:
  - do not promote `lista_blockdiag_ns200k_denseopt_sc3em3` or `lista_blockdiag_ns200k_denseopt_sc6em3` as paper baselines
  - mention them only as targeted positives on `multiwell_strong_transition` / `multiwell_strong_transition_hd` or as evidence that the dense optimizer does not transfer cleanly to block-diagonal LISTA
- Treat historical block-diagonal claims through the repaired March 17/19 MLP controls only; the old mislabeled controls remain invalid provenance and should not be reused.
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
- [follow-up `200k` paper summary](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/paper_benchmark_summary.md)
- [follow-up `200k` forecasting summary](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/forecasting_summary.md)
- [fair `200k` `generic_sparse` vs canonical `generic_sparse`](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/compare/vs_canonical_generic_sparse/generic_sparse_ns200k_best_vs_generic_sparse/forecasting_comparison.md)
- [promoted dense Stage 4 vs fair `200k` `generic_sparse`](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/compare/vs_generic_sparse_ns200k_best/lista_dense_promoted_stage4_vs_generic_sparse_ns200k_best/forecasting_comparison.md)
- [Stage-4 dense rerun summary](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/collect/paper_benchmark_summary.md)
- [Stage-4 dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/compare/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3_vs_generic_sparse/forecasting_comparison.md)
- [Kuramoto dimension summary](/home/mila/l/lia/skae/results/kuramoto_dimension_sweep_dt00625_200k_20260309/collect/kuramoto_dimension_summary.md)
- [Stage-1 dense-LISTA easy-system summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/collect/paper_benchmark_summary.md)
- [Kuramoto recovery summary](/home/mila/l/lia/skae/results/kuramoto_recovery_seq8_20260305/forecasting_summary.md)
- [intrinsic-HD `dt` rescue rerun summary](/home/mila/l/lia/skae/results/intrinsic_hd_dt_rescue_20260308_rerun1/forecasting_summary.md)
- Appendix-only `50k` references:
  - [v4 paper summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/paper_benchmark_summary.md)
  - [v4 final forecasting summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/forecasting_summary.md)
  - [v4 pass-2 `dt` resolution summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/dt_resolution/pass2/dt_resolution.md)
  - [dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_dense_vs_generic_sparse/forecasting_comparison.md)
  - [block-diagonal vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md)
