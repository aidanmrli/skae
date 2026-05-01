# Paper Experiment Evidence Map

Date: May 1, 2026

This is the paper-facing map for organizing the NeurIPS experiments section. It
compresses the live experiment record around the evidence chain in the draft:
multibasin Koopman learning motivates sparse supports, sparse supports produce
inspectable support objects, and those support objects should be useful for
routing prediction. Labels and known basin counts remain evaluation-only
signals on benchmark systems; they are not part of training-time method design
or deployment routing.

The experiments must keep two claims separate. First, a support can be a
static label: observing it can tell us which basin or coherent region contains
the state. Second, a support can be a dynamical selector: observing it can tell
us which active latent coordinates, which columns or subspaces of the learned
Koopman transition, or which centered local linear law should be used for
prediction. The first claim does not imply the second. A support may identify a
basin while the actual next-step dynamics depend on inactive coordinates,
continuous coefficient values, or cross-coordinate couplings. This is why the
paper first measures support agreement with basin labels and then separately
tests non-oracle support-routed prediction.

April 29 live recomputation note: the expanded table-refresh artifacts have
landed and the manuscript tables have been recomputed. The fixed-`17`
five-root forecasting source has `1261` rows; Table 1 interpretability has
`34,047` rows and `0` failures; Table 2 self-routed forecasting has `9,796`
rows and `0` failures; Table 3 support refresh has `189,708` rows and `0`
failures; Table 4 Dysts collected `894/900` requested rows, with the six
missing rows confined to LISTA-BD. The controlled multibasin tables strengthen
the paper. This note is superseded for the current Dysts display by the fully
collected `dt x30` packet: Dense MLP is worst at every horizon, LISTA is
aggregate-best at `H500`--`H1500` and significant versus Dense at every
horizon under the system-level aggregate Wilcoxon/Holm test, and LISTA-BD is
aggregate-best from `H2000` through `H5000` and significant from `H1000`
onward. Treat Dysts as an external forecasting/discretization-sensitivity
stress test, not as support-basin evidence.

April 30 verification note: the active Table 1 aggregator and the Tables 2-4
within-system paired-test script were rerun on compute allocation `9411806` and
reproduced the manuscript-facing counts. The draft tables are current with the
expanded seed artifacts.

April 30 Table 2 two-horizon/matched-dimension update: at the user's request,
Table 2 now reports both `H100` and `H1000` routed/global ratios side by side,
and the LISTA-SB row has been replaced by the matched `d_z=256` self-routing
artifact. The paired-test JSON and routing summary CSV were regenerated on
compute allocation `9420481` from the expanded five-model seed-`0`--`14`
self-routed artifact plus the matched `d_z=256` LISTA-SB packet. Displayed
point estimates are now cross-system IQMs of within-system finite-ratio IQMs.
The `[K/17]` counts use within-system Wilcoxon/Holm tests over all seed slots
with censored handling for invalid H-step routed/global comparisons. The
matched row weakens exact-support routing but preserves strong family-local
routing, especially at `H1000`.

April 30 matched Table 1/Table 3 completion note: the matched `d_z=256` Table
1 interpretability diagnostics and Table 3 support-refresh rows completed with
`0` failures. Interpretability jobs `9412879 -> 9412880` wrote `6,885` rows;
support-refresh jobs `9412881`--`9412883 -> 9412884` wrote `191,400` rows.
Tables 1--3 were recomputed on allocation `9420481` and the manuscript/PDF now
use the matched LISTA-SB row where available. Table 4 Dysts matching is
deferred to the later full-horizon run.

April 30 dense-K LISTA add-on queue note: a plain dense-K LISTA row is now
queued for Tables 1--3 under
[results/transition_rich_lista_dense_p256_hardinit_table123_20260430](/home/mila/l/lia/skae/results/transition_rich_lista_dense_p256_hardinit_table123_20260430).
The new `lista_dense_signsplit_p256_hardinit_basin_partition` task table has
`255` rows (`17` systems x `15` seeds) and matches the current LISTA Table
1--3 training settings: hard-init sampling, `d_z=256`, `sequence_length=8`,
`200000` optimization steps, sparsity coefficient `0.003`, and a dense Koopman
matrix with no soft-block or block-diagonal K regularizer. Launcher `9423747`
submitted training array `9423749`, collector `9423750`, Table 1
interpretability jobs `9423751 -> 9423752`, Table 2 self-routed jobs
`9423753`--`9423755 -> 9423756`, and Table 3 support-refresh jobs
`9423757`--`9423759 -> 9423760`. Do not revise the display plan until these
jobs complete and the table builders are rerun.

April 30 Table 3 redesign queue note: the main Table 3 display should be
rebuilt with period groups rather than a `Fallback` column. Period means the
number of Koopman rollout steps between decode/re-encode support refreshes
after controlled target-basin entry. The missing MLP-control refresh rows have
been queued under
[results/periodic_support_refresh_mlp_controls_seed0to14_20260430](/home/mila/l/lia/skae/results/periodic_support_refresh_mlp_controls_seed0to14_20260430):
Sparse MLP-BD shards `9423980`--`9423982`, Sparse MLP shards
`9423983`--`9423985`, Dense MLP shards `9423986`--`9423988`, merge `9423991`.
Table rebuild job `9423996` depends on this merge and dense-K LISTA refresh
merge `9423760`. Do not add the expanded Table 3 rows to the draft until the
dependent rebuild lands.

April 30 statistical-protocol documentation note: the manuscript appendix and
[EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md) now give the
repeatable testing procedure. For each table cell, construct paired deltas
within each system, run one-sided paired Wilcoxon tests inside systems,
Holm-correct the per-system p-values across the eligible systems for that
cell, and report `[K/N]` as the count of Holm-cleared systems. The point
estimate and the significance count are deliberately separate: IQM summarizes
finite observed values, while `[K/N]` summarizes within-system reproducibility.
For Table 2 only, seed slots `0`--`14` are retained with censored deltas for
invalid H-step routed/global comparisons; this is an evaluation convention for
evaluable versus unevaluable forecasts, not a causal diagnosis of learning
failure.

April 29 late higher-dimensional hard-system queue note: a focused sparse-KAE
forecasting redo is now in flight for fixed 8-basin CLV, Hopfield `N=16/P=16`,
and Kuramoto identical `N=16`. The purpose is to revisit the old hard-system
forecasting evidence under the shifted sparse Koopman-autoencoder narrative
with a clean tanh Dense MLP baseline instead of the old ReLU zero-sparsity
control. The submitted chain is training array `9412218`, collector `9412219`,
and comparisons `9412220`--`9412222` under
[results/hard_system_sparse_kae_redo_p1024_seq8_100k_halflr_sc6em3_tanh_dense_20260429](/home/mila/l/lia/skae/results/hard_system_sparse_kae_redo_p1024_seq8_100k_halflr_sc6em3_tanh_dense_20260429).
It uses `d_z=1024`, `sequence_length=8`, `100k` steps, half learning rates,
`15` seeds, and six recipes: Dense MLP, Sparse MLP, Sparse MLP-BD, LISTA,
LISTA-BD, and LISTA-SB. This is pending execution evidence; do not change the
hard-system claims until the collector and dense-tanh comparisons finish.

Use this file as the first stop for drafting. Use
[EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md) for detailed
experiment logs, [PAPER_TRACK_STATUS.md](/home/mila/l/lia/skae/docs/PAPER_TRACK_STATUS.md)
for high-level claim status and wrap-up priorities, and
[SUPPORT_OBJECT_GLOSSARY.md](/home/mila/l/lia/skae/docs/SUPPORT_OBJECT_GLOSSARY.md)
for exact support-object definitions.

## Recommended figures and tables

The experiments section should use figures and tables in the same order as the
reader's current chain: multibasin problem, supports as region indicators,
model-produced supports, and support-based routing. The main text should avoid
queue chronology and instead use each display item to close one link in that
chain.

### Main-text display order

1. **Figure 1: Support maps agree with basin structure.**
   Purpose: show that the model produces discrete support objects that align
   spatially with benchmark basins when labels are used only for evaluation.
   Recommended panels: three representative fixed-`17` systems as rows; true
   basin labels, learned exact or family support labels, top-`8` support-family
   labels, support entropy or fragmentation map, and support-versus-basin
   confusion matrix as columns. Use systems with complementary behavior, such
   as one clean polygonal/multiwell case, one gated local-linear case, and one
   transition-rich or boundary-stress case. Caption rule: explicitly state that
   basin labels are evaluation overlays, not training inputs.

2. **Table 1: Fixed-`17` support-label agreement and in-benchmark forecasting.**
   Purpose: quantify the first link of the chain. Columns should include
   `H100/H500/H1000`, `H(B|S_abs)`, `H(S_abs|B)`, `U_exact`, and freeze or robustness
   ratio if retained. Rows should separate sparse-latent Koopman, sparse MLP
   Koopman, and no-sparsity MLP controls. Keep standard-sampling rows separate
   from hard-init/near-boundary-sampling rows; do not mix them into one causal
   architecture table.
   *2026-04-30 matched-dimension LISTA-SB note.* The `d_z=256` LISTA-SB row now
   replaces the Table 1 LISTA-SB row. Forecasting remains strong versus Dense
   MLP under the same within-system Wilcoxon/Holm protocol: `H100 = 0.0198
   [15/17]`, `H500 = 0.0640 [15/17]`, and `H1000 = 0.0757 [15/17]`.
   Interpretability remains positive: `H(B|F_abs)=0 [13/13]`, wrong-support ratio
   `1.10e4 [13/13]` at `h=1`, `14.0 [11/13]` at `h=20`, and mean
   `|S_abs| = 99 [17/17]`. This is an averaged absolute-threshold active count,
   not the fixed top-`8` route size.

3. **Table 2: Non-oracle support-routed prediction.**
   Purpose: establish that the same support objects can select useful local
   predictors without oracle basin labels. This table is necessary because
   support-label agreement is only a static membership test; routing tests
   whether the support identifies useful latent coordinates or local laws for
   prediction. The main table should foreground exact top-`8` routing because
   it is the most deployment-like support object.
   Columns should report routed/global error ratio and win rate on all states
   and on states far from basin boundaries. Include the no-sparsity MLP control
   because it is the clearest negative routing comparator.
   *2026-04-28 statistics note, superseded by the seed-slot rule below.* Use
   within-system seed-paired Wilcoxon tests on `log(routed/global)` ratios,
   Holm-corrected across the `17` systems, and report `[K/17]` in the table.
   Cross-system sign counts may be used only as descriptive direction checks;
   they are not the manuscript-facing confirmatory statistic.
   *2026-04-30 two-horizon/matched-dimension update.* The expanded self-routed
   rows are complete with `9,796` five-model rows and `1,980` matched
   `d_z=256` LISTA-SB rows. The manuscript now shows `H100` and `H1000` side
   by side, with the LISTA-SB row replaced by the matched `d_z=256` artifact.
   Exact top-`8` routing is dimension-sensitive: matched LISTA-SB exact routes
   clear only `1/17`--`2/17` systems at `H100` and `7/17`--`8/17` at `H1000`.
   Family routing is the robust signal: matched LISTA-SB family-local clears
   `11/17`--`13/17` at `H100` and `15/17` at `H1000`, while LISTA-BD and sparse
   MLP controls also show strong family-local counts.
   *Repeatable Table 2 testing rule.* The displayed cell value is the
   cross-system IQM of within-system IQMs of finite positive
   routed/same-model-global ratios. The `[K/17]` count is computed separately
   over seed slots `0`--`14` inside each system. Finite
   pairs use `log10(routed/global)`. Finite routed with invalid global is a
   censored routed win, invalid routed with finite global is a censored routed
   loss, and both-invalid or missing seed slots are neutral. The censoring cap
   is chosen just outside the observed finite log-ratio range for the horizon.
   This keeps hard seeds in the test without calling invalidity a specific
   learning failure.
   *2026-04-30 matched-dimension LISTA-SB note.* The matched `d_z=256`
   self-routed row is now the manuscript Table 2 LISTA-SB row. It weakens
   exact-support routing but preserves strong family-local routing. At `H100`,
   exact support-gated/support-local clear only `1/17`--`2/17`, while
   family-local clears `11/17` all and `13/17` deep. At `H1000`, exact routes
   clear `7/17`--`8/17`, while family-local clears `15/17` on both all and
   deep slices. Interpret exact support identity as threshold- and
   dimension-sensitive; interpret support-family routing as the more robust
   local-partition result.
   *2026-05-01 control-table audit.* The same-protocol control pass completed
   under
   [results/transition_rich_table2_controls_20260430/self_routed_controls](/home/mila/l/lia/skae/results/transition_rich_table2_controls_20260430/self_routed_controls)
   with jobs `9425249`--`9425263 -> 9425264`, `12,245` rows, and `0`
   failures. The robust summary artifacts are
   [table2_partition_controls_h100_h1000_censored_seed15_summary.csv](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table2_partition_controls_h100_h1000_censored_seed15_summary.csv)
   and
   [table2_partition_controls_compact.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table2_partition_controls_compact.tex).
   This table supplies the intended control chain: oracle basin partitions ask
   whether local laws help when the benchmark partition is supplied,
   support-family selection asks whether the learned sparse representation
   provides a label-free substitute, and random/cluster partitions test whether
   generic subdivision explains the gain. The completed LISTA-SB control row is
   p64 while the main Table 2 row is matched `d_z=256`, so use the current
   table as appendix/audit evidence until the matched follow-up queued under
   [results/transition_rich_table2_controls_lista_sb_p256_20260501/self_routed_controls](/home/mila/l/lia/skae/results/transition_rich_table2_controls_lista_sb_p256_20260501/self_routed_controls)
   completes. That follow-up is shards `9429041`--`9429043`, with merge
   `9429044`.

4. **Figure 2: Support refresh and routing during basin transfer.**
   Purpose: make the routing mechanism visual rather than only tabular.
   Recommended panels: state-space trajectory with evaluation-only basin
   labels, inferred support object over time, refresh events, route-target
   fraction or fallback over time, and rollout comparison between global `K`,
   previous source-basin support, and refreshed-support routing. Caption
   rule: distinguish controlled-transfer evidence from a deployment-ready
   intervention/control claim.
   *2026-04-28 statistics note.* If this is kept as a table in the manuscript,
   the confirmatory statistic is a within-system paired Wilcoxon test over
   controlled transfer pairs (refreshed-support vs previous-support), with
   Holm correction across the `12` systems with transfer coverage.
   *2026-04-29 refresh note.* The expanded support-refresh pass is complete
   after a local merge, with `189,708` rows and `0` failures. Both LISTA
   variants and both refresh periods clear `12/12` systems under
   transfer-pair Wilcoxon/Holm. LISTA-SB has the strongest error ratios
   (`2.5e-3` at period `1`, `9.7e-4` at period `10`), while LISTA-BD also
   gives consistent improvement (`0.14` and `0.038`).
   *2026-04-30 matched-dimension update.* The matched `d_z=256` LISTA-SB
   refresh replacement is complete and now supplies the manuscript LISTA-SB
   row. It clears `12/12` systems at both periods; the within-system-IQM MSE
   ratios are `0.107 [0.085,0.142]` at period `1` and
   `0.031 [0.027,0.035]` at period `10`.

5. **Table 4: Dysts long-horizon forecasting.**
   Purpose: answer whether sparse-latent Koopman models remain competitive
   beyond the controlled low-dimensional multibasin benchmark.
   *2026-05-01 paper-facing update.* The main table now uses the fully collected
   `dt x30` sensitivity packet rather than the older tiny-step `H<=60000`
   stress-test packet. It reports
   `H100/H500/H1000/H1500/H2000/H3000/H4000/H5000` on the same `12`-system
   shortlist, with length-`10` training windows, `100k` optimization steps,
   and re-encode periods `{10,25,50,100,150,200}`. Cell point estimate is the
   cross-system IQM of per-system seed-IQMs. The bracketed in-cell count is a
   per-system reproducibility count: the number of systems whose
   within-system paired Wilcoxon test against Dense MLP clears Holm correction
   across the `12` systems. The model-vs-Dense inference should use the new
   aggregate companion table: one candidate/Dense seed-IQM ratio per system,
   one-sided Wilcoxon over the `12` log ratios, Holm-corrected across all
   model-horizon comparisons. Under that aggregate test, LISTA is significant
   versus Dense at every horizon, LISTA-BD is significant from `H1000`
   through `H5000`, Sparse MLP is significant from `H1000` through `H5000`,
   Sparse MLP-BD is significant from `H500` through `H5000`, and LISTA-SB
   does not survive all-comparison Holm correction. Keep this table after
   alignment and routing so it reads as external forecasting support, not as
   the central interpretability evidence.
   *2026-04-28 tiny-step provenance.* The older `H5000`--`H60000` packet remains
   useful as a stress-test diagnostic because it exposed unstable hard
   block-diagonal LISTA rollouts, but it should not be pooled with the coarser
   `dt x30` table.
   *2026-04-30 add-on sensitivity update.* The LISTA soft-block `d_z=256`
   Dysts row has landed as an appendix-only sensitivity with `180/180`
   completed rows. It reports `H5000`--`H30000`, not the current main
   manuscript `dt x30` `H100`--`H5000` horizon grid, so it is not pooled into
   \cref{tab:dysts_long}.
   *2026-04-29 refresh note.* The `long60` evaluation and collector have
   completed with `894/900` requested rows. Six LISTA-BD rows are missing or
   invalid (`dysts:DequanLi` seed `3` and `dysts:LorenzCoupled` seeds
   `1,2,4,8,9`), so paired tests use common completed seeds. The result is
   mixed: dense-transition LISTA is best by aggregate IQM at every horizon
   (`0.198` at `H5000`, `4.93` at `H60000`) but clears at most `2/12` systems;
   LISTA-BD has catastrophic finite errors and should not be framed as a Dysts
   win.

6. **Figure 3, optional main or appendix: Dysts long-horizon visual packet.**
   Purpose: show representative long-horizon rollouts and avoid a purely
   numeric forecasting section. The paper-facing aggregate trend should use the
   `dt x30` log-scale raw-IQM plot; the seed-`0` `H1000`--`H5000` all-model
   phase portraits can be used in an appendix or coauthor handoff if a qualitative
   Dysts visual is needed. If included in main text, keep it visually secondary
   to the basin/support and routing figures.

### Appendix or supplement displays

- **Appendix Table A1: Support object definitions and sensitivity.**
  Include `absolute:0.001`, `relative:0.1`, top-`8`, exact support, support
  family, and dominant group. This prevents the main text from collapsing
  thresholded identifiability, local-law diagnostics, and deployment routing
  into one overloaded "support" claim.
- **Appendix Table A2: Fixed-`17` benchmark system inventory.**
  List system name, basin count used for evaluation, basin-label source,
  whether it is clean away from basin boundaries, transition-rich, boundary-stress, or
  mechanism/falsification oriented, and which main figure/table uses it.
- **Appendix Table A3: Per-basin deep-slice robustness check.**
  Use the completed `interpretability_per_basin_deep_pass1` packets to repeat
  the Table 1 support/freeze diagnostics under a per-basin top-quartile deep
  slice. This appendix should be framed as coverage robustness for the
  wrong-support ablation, not as a replacement for the global
  deep-slice Table 1 numbers.
- **Appendix Table A4: Matched hard-init controls.**
  Summarize the near-boundary-sampling sparse MLP, block-diagonal sparse MLP,
  and no-shrink dense MLP controls. This is the guardrail that prevents a
  LISTA-only claim.
- **Appendix Table A5: Centered local-law, zero-intercept, true-geometry, and
  random-partition diagnostics.**
  Use this to show the calibrated mechanism story: centered local laws are
  useful, zero-intercept local laws fail, true-geometry recovery is mixed, and
  support families are not merely random partitions.
- **Appendix Figure A1: Operator geometry diagnostic.**
  Only include in the main text if the paper claims geometric recovery. With
  the current mixed result, this is better as a falsification/limitation figure
  comparing learned local operators, true Jacobians/eigendirections, basin
  baselines, and random partitions.
- **Appendix Table A6: Fair `200k`, hard-system, and historical-provenance
  benchmark summary.**
  Keep the repaired `200k` cross-system benchmark, Kuramoto/Hopfield
  hard-system results, dense recipe-selection provenance, and `50k` audit in
  one compact appendix area rather than spreading them through the main
  experiments narrative. Once collected, add the queued `d_z=1024`,
  tanh-dense-baseline higher-dimensional redo here as a corrected hard-system
  sensitivity rather than as a main evidence pillar.

## 1. Do sparse supports align with basins when labels are used only for evaluation?

Paper role:
- This is the first main result. It tests whether the learned support object is
  an interpretable indicator of basin membership after fully label-free
  training.
- It is deliberately a static test. It asks whether the support says where the
  state is, not whether the support already proves which latent coordinates or
  local operator should advance the state.

Current result:
- On the fixed `17`-system multibasin benchmark, absolute-threshold supports
  on states far from basin boundaries rarely mix basin labels for the main
  sparse-latent models. The strongest exact-support agreement row is the
  sparse-latent Koopman autoencoder with a
  soft-block transition: `H(B|S_abs)=0.0000`, `H(S_abs|B)=0.0543`, and
  `U_exact=0.9923`.
- The sparse-latent Koopman autoencoder with a block-diagonal transition has
  stronger forecasting on the same table but more support fragmentation:
  `H(B|S_abs)=0.0000`, `H(S_abs|B)=0.3219`, and `U_exact=0.9646`.
- On the same fixed-`17` table, the short-to-long multibasin forecasting
  companion numbers are `H100/H500/H1000 = 0.0182 / 0.0491 / 0.0516` for the
  block-diagonal sparse-latent Koopman model, `0.0253 / 0.0719 / 0.0768` for
  the soft-block sparse-latent Koopman model, `0.0297 / 0.0614 / 0.0608` for
  the sparse MLP Koopman control, and `0.0433 / 0.0932 / 0.0911` for the dense
  non-sparse MLP Koopman control.
- The causal claim should be "induced sparse latent structure supports
  support objects that agree with basin labels," not "LISTA alone is
  responsible." The
  implementation makes this a latent-magnitude penalty claim: `GenericKM` forms
  the sparsity term as the mean latent `L1` norm and adds
  `SPARSITY_COEFF * sparsity_loss` to the objective
  ([model.py](/home/mila/l/lia/skae/skae/model.py:1173),
  [model.py](/home/mila/l/lia/skae/skae/model.py:1278)). The fixed-`17`
  manifest makes the sparse-versus-no-sparsity comparison explicit: sparse MLP
  controls use `generic_sparse` with `sparsity_coeff=0.003`, while no-shrink
  controls use `generic_no_shrink` with `sparsity_coeff=0.0`
  ([transition_rich_basin_partition_manifest.py](/home/mila/l/lia/skae/skae/benchmarks/transition_rich_basin_partition_manifest.py:1177),
  [transition_rich_basin_partition_manifest.py](/home/mila/l/lia/skae/skae/benchmarks/transition_rich_basin_partition_manifest.py:1251)).
- Sparse MLP controls are themselves competitive: under the standard-sampling
  control row in the locked final packet, the sparse MLP already has supports
  that do not mix basin labels on the states-far-from-boundaries slice and
  competitive forecasting (`H(B|S_abs)=0.0000`,
  `H(S_abs|B)=0.2449`, `U_exact=0.9772`, `H(B|F_abs)=0.0000`, and
  `H100/H500/H1000 = 0.0297 / 0.0614 / 0.0608`)
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:12)).
  The dense hard-init sparse-latent Koopman finalist gives the cleanest exact
  support-compression result (`H(S_abs|B)=0.0543`, `U_exact=0.9923`), but that row
  is mixed-regime because the promoted LISTA roots use hard-init oversampling
  while the MLP rows in the locked packet use standard sampling
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:6),
  [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:14)).
- The matched near-boundary-sampling control packet is an important causal
  guardrail. Under the same hard-init sampling regime, sparse MLP controls are
  almost tied on the selected deep slice (`H(B|S_abs)=0.0000`,
  `H(S_abs|B)=0.2068`, `U_exact ~= 0.98`) and forecast strongly
  (`H100/H500/H1000 = 0.0082 / 0.0260 / 0.0273` for plain sparse MLP and
  `0.0082 / 0.0252 / 0.0264` for block-diagonal sparse MLP), while the
  no-shrink dense MLP control is much worse functionally
  (`0.5704 / 2.6733 / 3.8044`)
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1/transition_rich_final_comparison.md:10)).

Interpretation:
- Away from basin boundaries, sparse supports often identify basin structure.
  The direct evidence is that the locked final comparison is explicitly on the
  `absolute:0.001` / states-far-from-boundaries support slice and all sparse
  rows in that table have `H(B|S_abs)=0.0000`, meaning that an absolute-threshold support object does
  not mix basin labels on that slice
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:3),
  [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:12)).
  The same rows have low fragmentation or high dominant-support mass
  (`H(S_abs|B)=0.0543`, `U_exact=0.9923` for the dense sparse-latent finalist;
  `H(S_abs|B)=0.2449`, `U_exact=0.9772` for the standard sparse MLP control), and
  support families saturate (`H(B|F_abs)=0.0000`). The matched hard-init sparse MLP
  controls replicate the support-label agreement read under the oversampled regime
  (`H(B|S_abs)=0.0000`, `H(S_abs|B)=0.2068`, `U_exact ~= 0.98`, `H(B|F_abs)=0.0000`)
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1/transition_rich_final_comparison.md:3),
  [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1/transition_rich_final_comparison.md:10)).
- Exact support identity is not a global invariant. Near separatrices and weak
  coordinates, support families or fixed-size top-`k` supports are more robust
  than a single absolute threshold.
- This agreement result is necessary but insufficient. It says that supports
  carry information about basin membership, but it does not say that the active
  support selects useful columns of the learned transition or a useful centered
  local law. That is the role of the non-oracle routing experiments.
- The no-shrink control is a useful negative/guardrail mainly on functional
  robustness, not on every entropy metric: it can still show thresholded
  supports that do not mix basin labels on the selected states-far-from-boundaries
  slice, but it is much worse in
  hard-init forecasting and freeze robustness. Therefore the interpretation
  should emphasize induced sparsity plus functional stability, not only one
  entropy column.

Project implication:
- The paper should write this as support agreement with basin labels, not
  basin-block alignment and not oracle clustering. Basin labels are only used
  to score support purity, fragmentation, and dominant-support mass.
- Standard-sampling and near-boundary-sampling rows should stay separate in
  tables. The hard-init controls support induced sparsity more than a
  LISTA-specific architecture claim.
- Table 1 should therefore have one row family for sparse-latent Koopman, one
  for sparse MLP Koopman, and one for no-shrink / no-sparsity MLP controls, with
  standard-sampling and hard-init rows visibly separated. The main prose should
  say that latent sparsity can produce support objects that agree with basin
  labels, while the strongest exact-support compression belongs to the dense
  sparse-latent finalist and the strongest architecture-isolating hard-init
  evidence belongs to the broader induced-sparsity claim.

Next drafting step:
- Build the main interpretability figure: true basin labels, learned
  support/support-family labels, support entropy, and a support-versus-basin
  confusion matrix on representative systems.

Primary artifacts:
- [FIXED17_LISTA_RESULTS_INDEX.md](/home/mila/l/lia/skae/docs/FIXED17_LISTA_RESULTS_INDEX.md)
- [transition_rich_basin_support_metric_definitions.md](/home/mila/l/lia/skae/docs/transition_rich_basin_support_metric_definitions.md)
- [results/transition_rich_hardinit_mlp_controls_seed10_20260416](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416)
- [EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md), fixed-`17`
  support-label agreement entries

## 2. Do those same supports select useful local predictors without oracle basin labels?

Paper role:
- This is the functional test of the support-as-dynamical-selector claim. The first
  experiment asks whether learned supports align with basin labels when labels
  are used only for evaluation. This second experiment asks whether the same
  model-produced support objects do useful dynamical work: can a support
  inferred from the current latent state choose a better local predictor than
  applying one learned global latent transition everywhere?
- In the manuscript logic up to the experiments section, this closes the
  second arrow in the chain
  `sparse latent code -> support agrees with basin labels -> support-routed local
  prediction`. Without this result, support alignment could be only a static
  clustering observation. With it, the supports become evidence for a
  label-free routing variable that selects local Koopman-style prediction.
- This is the counterfactual the reader needs in mind: a support can label a
  basin well and still fail dynamically if the coordinates that mark basin
  membership are not the coordinates whose transition columns carry the local
  dynamics.

What is being tested:
- Freeze the trained model and infer a support object from the latent state;
  do not use true basin labels, known basin counts, or trajectory-to-basin
  assignments for routing.
- Use the inferred support in two distinct prediction diagnostics:
  support-gated prediction, where the same learned global transition is
  applied after masking the centered latent input by the current support, and
  centered local-law prediction, where the current support or support family
  selects a post-hoc centered local operator fitted after model training.
- Evaluate on disjoint forecast trajectories and compare routed prediction
  against the model's learned global transition. Ratios below `1` and high win
  rates mean the support selected a useful local predictor.
- Foreground fixed-size exact top-`8` supports because they are the most
  deployment-like object: every state gets a route key without choosing an
  activity threshold.

Terminology clarification:
- "Routing" should not be written as if the trained model contains separate
  learned Koopman matrices per support. The base trained Koopman autoencoder
  has one learned global transition.
- Support-gated prediction is support-conditioned masking of the global
  transition, not switching among independently trained local Koopman matrices.
  Algebraically, the support selects which centered latent coordinates drive
  the transition, so it is closest to testing whether the support selects useful
  columns or subspaces of the learned global transition.
- Centered local-law prediction is a literal local-law routing diagnostic. It
  freezes the trained encoder and decoder, fits centered local slopes after
  training, and then uses the current model-produced support or family as the
  key that selects which local slope to apply.
- These local laws are not used during model training. They are post-hoc
  inference/evaluation objects. In the full non-oracle rollout packet, however,
  they are actually used to produce forecasts in the centered support-local and
  centered family-local modes; they are not merely scored passively.

Centered local-law mechanism packet:
- This is a one-step latent-dynamics diagnostic, not an autonomous forecasting
  study. For each trained checkpoint, generate benchmark trajectories, encode
  each state, and form held-out one-step latent pairs `(z_t, z_{t+1})`.
- Label each transition by the current basin, exact support, or support family.
  Basin labels are included as an evaluation/control partition only, not as a
  training or deployment router.
- Split transitions into fit/test subsets. For each partition class `c` with
  enough transitions, fit a ridge-regularized centered local law
  `z_{t+1} ~= center_c + A_c (z_t - center_c)`.
- Compare held-out one-step latent MSE against the learned global transition, one
  global centered latent slope, count-matched random partitions, and sometimes
  latent-k-means controls. Also evaluate support-gated versions of the learned
  transition using support prototypes or block-union masks.
- This packet tests mechanism: whether support/family partitions behave like
  local regime labels for one-step latent dynamics, and whether masking the
  learned global transition by support exposes useful local subspaces. It does
  not test full autonomous forecasting.

Full non-oracle rollout packet:
- This is the deployment-facing forecasting diagnostic. For each trained
  checkpoint, use one generated trajectory set to fit support/family centers
  and post-hoc local operators, and a disjoint trajectory set for held-out
  autonomous forecasts.
- Start each forecast from only the initial observation: encode `x_0` to
  `z_0`, then roll forward to horizons `H100`, `H500`, and `H1000`.
- In the global-transition baseline, forecast with the base trained transition.
- In support-gated prediction, infer the support from the current predicted
  latent state, then apply the learned global transition to the centered masked
  latent state. If the required support center is unavailable, fall back to the
  global transition.
- In centered local-law prediction, infer the exact support from the current
  predicted latent state, select the post-hoc local slope for that support, and
  apply it around the support center. In the family-level version, first map
  the exact support to a support family and then select the family-level local
  law. Missing routes fall back to the global transition.
- The route is recomputed from the model's own predicted latent state during
  rollout, not from the true future state and not from an oracle basin label.
  This is the packet that tests whether the supports can actually be used for
  inference-time forecasting.

Why this matters:
- The paper's premise is not merely that sparse supports can be inspected after
  training. The premise is that supports can identify which local linear law is
  relevant for the current state.
- Support agreement with basin labels alone cannot establish this. A support
  might identify a basin but remain dynamically useless, or it might only
  reproduce an evaluation label without improving prediction.
- Non-oracle routing is therefore the necessary bridge from interpretability to
  use: it tests whether the support object produced by the model can replace
  an unavailable oracle basin label when selecting local prediction behavior.

Current result:
- The non-oracle self-routed forecasting packet is complete with `510/510`
  runs, `24,600` rows, and `0` failures.
- The strongest current exact-support result uses fixed-size top-`8` supports
  for the dense sparse-latent Koopman model. At `H1000`, support-gated
  prediction has all-slice median ratio / win rate `0.228 / 0.920` against the
  learned global transition, and centered support-local prediction has
  `0.275 / 0.947`.
- On deep states, the centered local-law route is even stronger:
  ratio / win rate `0.207 / 0.985`.
- The dense non-sparse MLP control is much weaker under the same exact top-`8`
  routing rule: `0.924 / 0.539` for support-gated prediction and
  `1.000 / 0.496` for centered local laws.
- The direct periodic-support-refresh ablation supports the rollout mechanism
  after target-basin entry for dense sparse-latent exact top-`8` supports.
  Refreshed-support gating reaches route-target fraction `0.8552 / 0.8886`,
  fallback `0.1392 / 0.1058`, and refreshed-versus-previous-support MSE ratio
  `0.0093 / 0.0131` for re-encode periods `1 / 10`. Support-family routing is cleaner, while
  block-diagonal exact supports remain weaker.

Interpretation:
- The same learned support objects can route useful predictors without oracle
  basin labels.
- The safest deployment-facing support object is currently fixed-size top-`8`,
  not an absolute-threshold exact mask.
- The periodic-refresh claim should be narrowed: exact-support evidence is
  strongest for dense sparse-latent top-`8` supports after basin entry; support
  families support a broader version.

Project implication:
- The experiments section should foreground exact top-`8` self-routing before
  broader support-family summaries. Family-level results help robustness, but
  they are less specific to induced sparsity because some controls also form
  clean families.
- The paper should split the routing claim into two questions: does the support
  identify useful columns/subspaces of the learned global transition, and does the
  support work as a non-oracle key for post-hoc centered local laws? These are
  related but not the same claim.
- Be explicit that the main trained model still uses one global transition. The
  masking and post-hoc local-law modes are inference-time/evaluation variants,
  not training-time supervision or oracle routing.

Next drafting step:
- Add a routing table and a controlled-transfer/refresh figure showing
  state-space path, true basin label for evaluation, inferred support object,
  refresh events, route availability, and post-transfer forecast error.
- TODO: Add or clearly label an inference-time forecasting ablation that uses
  the post-hoc centered support-local or family-local laws as a prediction
  rule. This does not require using local laws during training; the question is
  whether a frozen sparse encoder/decoder plus support-selected post-hoc local
  operators improves held-out forecasting relative to the global transition and
  support-gated global transition.

Primary artifacts:
- [results/transition_rich_self_routed_forecasting_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420)
- [results/periodic_support_refresh_fixed17_seed0_20260425](/home/mila/l/lia/skae/results/periodic_support_refresh_fixed17_seed0_20260425)
- [controlled_transfer_switching_experiment_20260423.md](/home/mila/l/lia/skae/docs/planning/controlled_transfer_switching_experiment_20260423.md)

## 3. Do sparse-latent Koopman models remain competitive for long-horizon forecasting?

Paper role:
- This is the external forecasting stress test. It shows the sparse-latent
  models are not only interpretable on the controlled multibasin benchmark.

Current result:
- *2026-04-29 refresh:* the `seq_len=10`, up-to-`15`-seed, `12`-system
  `H{5K..60K}` Dysts rerun has been collected under
  [results/dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428/collect](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428/collect).
  The collector has `894/900` requested rows. The six missing or invalid rows
  are all LISTA-BD (`dysts:DequanLi` seed `3` and `dysts:LorenzCoupled` seeds
  `1,2,4,8,9`), and within-system paired tests use common completed seeds.
- Dense-transition LISTA has the best cross-system IQM at every reported
  horizon: `0.198`, `1.02`, `2.88`, `3.81`, `4.30`, `4.65`, and `4.93` for
  `H5000` through `H60000`. However, the within-system Holm-corrected counts
  are weak (`2/12`, `2/12`, `0/12`, `0/12`, `2/12`, `0/12`, `0/12`), so this
  should be described as aggregate competitiveness rather than broad
  systemwise dominance.
- LISTA-BD is not a positive Dysts result in this rerun. Its aggregate IQM is
  dominated by catastrophic finite errors (`7.9e29` at every displayed
  horizon) and it clears `0/12` systems against Dense MLP. Sparse MLP and
  Sparse MLP-BD stay near the Dense MLP baseline and clear at most `1/12`
  systems at the longest horizons.
- *2026-05-01 `dt x30` sensitivity:* the full `12`-system coarser-step packet
  completed under
  [results/dysts_dt30_basinblock_p256_seq10_100k_20260430/long_horizon_eval/collect](/home/mila/l/lia/skae/results/dysts_dt30_basinblock_p256_seq10_100k_20260430/long_horizon_eval/collect)
  with `1080/1080` rows, `0` pending/invalid rows, and median full-horizon
  finite coverage of `1` for every root at every reported horizon. The
  paper-facing rebuild now reports cross-system IQMs of per-system seed-IQMs:
  at `H5000`, LISTA-BD is `0.5636`, LISTA is `0.5740`, Sparse MLP-BD is
  `0.6284`, Sparse MLP is `0.6433`, LISTA-SB is `0.9691`, and Dense MLP is
  `1.0605`. LISTA is best at `H500`--`H1500`, LISTA-BD is best from `H2000`
  through `H5000`, and Sparse MLP-BD is best at `H100`.
- The within-system Wilcoxon/Holm read against Dense MLP is positive but not
  universal. LISTA clears `3/12` systems at `H100`, `6/12` at `H500`, and
  `5/12` at `H1000`--`H5000`; LISTA-BD clears `2/12`, `2/12`, `4/12`,
  `4/12`, `4/12`, `5/12`, `4/12`, and `3/12` from `H100` through `H5000`;
  Sparse MLP-BD peaks at `7/12` at `H1500`; LISTA-SB clears `0/12`.
  The paper-facing raw-MSE trend uses log scale because the displayed IQM MSEs
  span about `5.6e3x`; linear-scale and side-by-side scale-check plots are
  retained as diagnostics. Trend bands are fixed-system seed-bootstrap `95%`
  intervals around the cross-system IQM, so they quantify finite-seed
  uncertainty rather than variation across the `12` systems. The table and
  plots are under
  [docs/figures/neurips_paper_2026](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026),
  and phase portraits for all `12` systems at `H1000`--`H5000` are under
  [docs/figures/dysts_dt30_phase_portraits_seed0_h1000_h5000_all_models_20260501](/home/mila/l/lia/skae/docs/figures/dysts_dt30_phase_portraits_seed0_h1000_h5000_all_models_20260501).
- A separate LISTA soft-block `d_z=256` Dysts sensitivity row has completed
  under
  [results/dysts_long_horizon_eval_dysts_seq10_lista_softblock_p256_sc6em3_seeds0to14_20260428](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_dysts_seq10_lista_softblock_p256_sc6em3_seeds0to14_20260428).
  It covers the same `12` Dysts systems and seeds `0`--`14` with `180/180`
  completed evaluations, but the available collector is the older
  `H5000`--`H30000` horizon grid. Treat it as appendix sensitivity, not as a
  row in the main `H5000`--`H60000` Dysts table.
- The older seven-root `H<=30000` comparison is now superseded for the
  manuscript table. Keep it only as provenance for older visual packets or
  exploratory discussion.
- The older fair `200k` cross-system benchmark remains useful supporting
  context at `H100-H3000`. It is a three-way read rather than a simple
  sparse-versus-dense result: sparse MLP is best at `H100/H1000`
  (`2.947e-4 / 0.0240`), promoted dense sparse-latent Koopman is best at
  `H500` and `H1500-H3000` (`0.0047 / 0.0449 / 0.0627 / 0.0880 / 0.1039`),
  and zero-sparsity improves late-horizon coverage on more systems than it
  wins by median.

Interpretation:
- Sparse-latent Koopman models remain competitive in aggregate on Dysts, and
  under the `dt x30` sensitivity they usually improve over the Dense MLP
  baseline. The per-system Holm counts are moderate rather than universal, so
  this remains forecasting support rather than a broad dominance claim.
- Stronger transition structure is fragile in the older tiny-step packet but
  not in the `dt x30` packet. The correct paper phrasing is sensitivity to the
  discrete-time benchmark design, not categorical block-diagonal failure.
- The `dt x30` sensitivity changes that diagnosis from "hard block-diagonal
  LISTA is categorically weak on Dysts" to "the old tiny-step/very-long
  composition benchmark exposes instability." Because the sensitivity changes
  the timestep and horizon index, it should be labeled as a
  benchmark-discretization diagnostic rather than pooled into the old
  `H5000`--`H60000` table.
- This benchmark is supporting evidence, not a support-label agreement test,
  because Dysts is not the controlled labeled multibasin benchmark.

Project implication:
- Put Dysts after the multibasin alignment and routing results. It should
  answer "do we remain competitive?" rather than replace the central
  interpretability story.
- Use the fair `200k` benchmark as supporting context if the paper needs a
  shorter-horizon cross-system table, but do not let it reorder the main
  experiments section.

Next drafting step:
- Keep the Dysts table concise and explicitly secondary. If a Dysts visual is
  included, use the `dt x30` log-scale raw-IQM trend as the main view and keep
  the ratio-to-Dense or linear-scale plots as supporting diagnostics rather than
  replacing the statistical table.

Primary artifacts:
- [results/dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428/collect](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428/collect)
- [results/dysts_seq10_lista_softblock_p256_sc6em3_seeds0to14_20260428](/home/mila/l/lia/skae/results/dysts_seq10_lista_softblock_p256_sc6em3_seeds0to14_20260428)
- [results/dysts_long_horizon_eval_20260414](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_20260414)
- [results/dysts_long_horizon_eval_mlp_blockdiag_20260415](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_mlp_blockdiag_20260415)
- [docs/figures/dysts_phase_portraits/dysts_h30000_best_root_phase_portraits_manifest.json](/home/mila/l/lia/skae/docs/figures/dysts_phase_portraits/dysts_h30000_best_root_phase_portraits_manifest.json)
- [results/paper_zero_sparse_benchmark_200k_20260321](/home/mila/l/lia/skae/results/paper_zero_sparse_benchmark_200k_20260321)

## 4. Other useful things

Keep these as supporting, appendix, or falsification evidence unless the paper
needs a specific response to a reviewer-style objection.

- Centered local-law mechanism: support-conditioned centered local laws beat
  the learned global transition on states far from basin boundaries, with win rates
  `93.1%` for block-diagonal sparse-latent Koopman, `98.6%` for dense
  sparse-latent Koopman, and `100%` for the dense non-sparse MLP control on
  the deepest slice. This supports the local-dynamics framing but is not
  LISTA-specific.
- True local geometry: the second-audited true-Jacobian/eigendirection packet
  is mixed. LISTA support families often beat random partitions near
  attractors, but the dense non-sparse MLP can have lower absolute projected
  Jacobian error because its latent representation is closer to the identity
  map. Use this as a
  falsification diagnostic, not a headline claim.
- Controlled transfer: dense sparse-latent exact top-`8` supports switch well
  after a deliberate state-space bridge, while family-level switching is strong
  for multiple roots. This supports support switching but should be written
  separately from a deployment-ready intervention/control claim.
- Negative zero-intercept operator selection: the earlier affine-only local-law
  result failed even with oracle basin fits. Keep it because it justifies the
  centered local-law diagnostic and prevents overclaiming exact operator recovery.
- Phase portraits and visual packets: useful for senior-coauthor handoff and
  appendix figures, but not substitutes for the three main evidence questions.
- Hard-system forecasting: the Kuramoto/Hopfield and step-size-rescue packets
  are decision-grade limitation/support evidence. They show that step size is
  a real bottleneck, that block-diagonal structure gives targeted Kuramoto
  advantages at moderate dimension, and that Hopfield remains mostly
  MLP-better. This belongs in limitations or appendix unless the main paper
  needs a hard-system stress-test paragraph. A new `d_z=1024` hard-system redo
  with a tanh Dense MLP baseline is queued; treat it as pending until
  `9412219` collection and `9412220`--`9412222` comparisons finish.
- Label-free clustering and broad support-audit diagnostics: keep these as
  appendix/provenance for why literal binary support uniqueness is not enough.
  They explain known negatives such as Kuramoto, mixed continuous-separation
  cases such as Hopfield, and the shift toward support families, top-`k`
  supports, and support-label agreement on the fixed-`17` systems.
- Historical benchmark provenance: dense-LISTA recipe selection, the symmetric
  `50k` audit, invalidated historical block-`K` controls, and March queue
  repair notes should stay as provenance. The main text should cite the
  repaired `200k` and fixed-`17` packets instead.
- Queue history, failed/superseded packets, tuning provenance, and selector
  sensitivity should be linked from the detailed log or archive rather than
  repeated in the main experiment narrative.

Primary artifacts:
- [results/transition_rich_centered_chart_mechanism_20260420](/home/mila/l/lia/skae/results/transition_rich_centered_chart_mechanism_20260420)
- [results/true_jacobian_geometry_fixed17_seed0_20260424_reaudit](/home/mila/l/lia/skae/results/true_jacobian_geometry_fixed17_seed0_20260424_reaudit)
- [results/controlled_transfer_switching_fixed17_seed0_20260424_reaudit](/home/mila/l/lia/skae/results/controlled_transfer_switching_fixed17_seed0_20260424_reaudit)
- [results/kuramoto_dimension_sweep_dt00625_200k_20260309](/home/mila/l/lia/skae/results/kuramoto_dimension_sweep_dt00625_200k_20260309)
- [results/hard_system_sparse_kae_redo_p1024_seq8_100k_halflr_sc6em3_tanh_dense_20260429](/home/mila/l/lia/skae/results/hard_system_sparse_kae_redo_p1024_seq8_100k_halflr_sc6em3_tanh_dense_20260429)
- [PAPER_TRACK_STATUS.md](/home/mila/l/lia/skae/docs/PAPER_TRACK_STATUS.md), broad support-audit and label-free clustering entries
- [EXPERIMENTS_ARCHIVE.md](/home/mila/l/lia/skae/docs/EXPERIMENTS_ARCHIVE.md)
