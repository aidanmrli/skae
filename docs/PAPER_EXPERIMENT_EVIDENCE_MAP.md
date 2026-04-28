# Paper Experiment Evidence Map

Date: April 28, 2026

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
   `H100/H500/H1000`, `H(B|S)`, `H(S|B)`, `U_exact`, and freeze or robustness
   ratio if retained. Rows should separate sparse-latent Koopman, sparse MLP
   Koopman, and no-sparsity MLP controls. Keep standard-sampling rows separate
   from hard-init/near-boundary-sampling rows; do not mix them into one causal
   architecture table.

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

4. **Figure 2: Support refresh and routing during basin transfer.**
   Purpose: make the routing mechanism visual rather than only tabular.
   Recommended panels: state-space trajectory with evaluation-only basin
   labels, inferred support object over time, refresh events, route-target
   fraction or fallback over time, and rollout comparison between global `K`,
   stale/frozen source support, and refreshed-current support routing. Caption
   rule: distinguish controlled-transfer evidence from a deployment-ready
   intervention/control claim.

5. **Table 3: Dysts long-horizon forecasting.**
   Purpose: answer whether sparse-latent Koopman models remain competitive
   beyond the controlled low-dimensional multibasin benchmark.
   *2026-04-28 update — pending re-run.* Columns are being expanded from
   `H5000/H10000/H20000/H30000` to a single sweep
   `H5000/H10000/H20000/H30000/H40000/H50000/H60000`
   (single rollout to `H=60000` with periodic re-encoding; MSE is read at each
   horizon). This is explicitly a forecasting-beyond-training-source-horizon
   test: training uses windows of length `10` from 30K-step Dysts source
   trajectories, while evaluation uses a separate `long60` held-out test cache.
   The Dysts system list is narrowed from `15` to `12` (drops `Duffing`,
   `SprottTorus`, `RikitakeDynamo`) to save compute. The narrowed `5`-recipe
   paper-facing Dysts set is being re-trained at `sequence_length=10` (was `8`)
   with `15` seeds (was `10`) so per-system paired Wilcoxon + Holm at
   `alpha/12` has exact-test floor headroom. Re-encode periods are reduced
   from the prior `13`-element grid to
   `{50, 75, 100, 200, 400, 600, 1000}`. Cell point estimate is the IQM of
   per-system IQM-over-seeds; primary in-cell stat is per-system paired
   Wilcoxon vs Dense MLP, Holm-corrected over the `12` systems; companion
   sign-test on per-system IQM-deltas reported in the body text. Keep this
   table after alignment and routing so it reads as external forecasting
   support, not as the central interpretability evidence.

6. **Figure 3, optional main or appendix: Dysts long-horizon visual packet.**
   Purpose: show representative long-horizon rollouts and avoid a purely
   numeric forecasting section. Use the all-roots `H30000` best-root packet for
   appendix or a compact main-text montage if space permits. If included in
   main text, keep it visually secondary to the basin/support and routing
   figures.

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
- **Appendix Table A3: Matched hard-init controls.**
  Summarize the near-boundary-sampling sparse MLP, block-diagonal sparse MLP,
  and no-shrink dense MLP controls. This is the guardrail that prevents a
  LISTA-only claim.
- **Appendix Table A4: Centered local-law, zero-intercept, true-geometry, and
  random-partition diagnostics.**
  Use this to show the calibrated mechanism story: centered local laws are
  useful, zero-intercept local laws fail, true-geometry recovery is mixed, and
  support families are not merely random partitions.
- **Appendix Figure A1: Operator geometry diagnostic.**
  Only include in the main text if the paper claims geometric recovery. With
  the current mixed result, this is better as a falsification/limitation figure
  comparing learned local operators, true Jacobians/eigendirections, basin
  baselines, and random partitions.
- **Appendix Table A5: Fair `200k`, hard-system, and historical-provenance
  benchmark summary.**
  Keep the repaired `200k` cross-system benchmark, Kuramoto/Hopfield
  hard-system results, dense recipe-selection provenance, and `50k` audit in
  one compact appendix area rather than spreading them through the main
  experiments narrative.

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
  soft-block transition: `H(B|S)=0.0000`, `H(S|B)=0.0543`, and
  `U_exact=0.9923`.
- The sparse-latent Koopman autoencoder with a block-diagonal transition has
  stronger forecasting on the same table but more support fragmentation:
  `H(B|S)=0.0000`, `H(S|B)=0.3219`, and `U_exact=0.9646`.
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
  competitive forecasting (`H(B|S)=0.0000`,
  `H(S|B)=0.2449`, `U_exact=0.9772`, `H(F|B)=0.0000`, and
  `H100/H500/H1000 = 0.0297 / 0.0614 / 0.0608`)
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:12)).
  The dense hard-init sparse-latent Koopman finalist gives the cleanest exact
  support-compression result (`H(S|B)=0.0543`, `U_exact=0.9923`), but that row
  is mixed-regime because the promoted LISTA roots use hard-init oversampling
  while the MLP rows in the locked packet use standard sampling
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:6),
  [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:14)).
- The matched near-boundary-sampling control packet is an important causal
  guardrail. Under the same hard-init sampling regime, sparse MLP controls are
  almost tied on the selected deep slice (`H(B|S)=0.0000`,
  `H(S|B)=0.2068`, `U_exact ~= 0.98`) and forecast strongly
  (`H100/H500/H1000 = 0.0082 / 0.0260 / 0.0273` for plain sparse MLP and
  `0.0082 / 0.0252 / 0.0264` for block-diagonal sparse MLP), while the
  no-shrink dense MLP control is much worse functionally
  (`0.5704 / 2.6733 / 3.8044`)
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1/transition_rich_final_comparison.md:10)).

Interpretation:
- Away from basin boundaries, sparse supports often identify basin structure.
  The direct evidence is that the locked final comparison is explicitly on the
  `absolute:0.001` / states-far-from-boundaries support slice and all sparse
  rows in that table have `H(B|S)=0.0000`, meaning that a support object does
  not mix basin labels on that slice
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:3),
  [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:12)).
  The same rows have low fragmentation or high dominant-support mass
  (`H(S|B)=0.0543`, `U_exact=0.9923` for the dense sparse-latent finalist;
  `H(S|B)=0.2449`, `U_exact=0.9772` for the standard sparse MLP control), and
  support families saturate (`H(F|B)=0.0000`). The matched hard-init sparse MLP
  controls replicate the support-label agreement read under the oversampled regime
  (`H(B|S)=0.0000`, `H(S|B)=0.2068`, `U_exact ~= 0.98`, `H(F|B)=0.0000`)
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
  Refreshed-current support gating reaches target-support dominance
  `0.8319 / 0.8662`, route-target fraction `0.8552 / 0.8886`, fallback
  `0.1392 / 0.1058`, and refreshed-versus-frozen MSE ratio `0.0093 / 0.0131`
  for re-encode periods `1 / 10`. Support-family routing is cleaner, while
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
- *2026-04-28: prior `15`-system / `n=10`-seed / `seq_len=8` /
  `H{5K,10K,20K,30K}` Dysts results are being superseded by a re-run with
  `12` systems, `n=15` seeds, `seq_len=10`,
  `H{5K..60K}` (single rollout to `H=60000` using a separate `long60`
  held-out test cache), and reduced re-encode periods
  `{50, 75, 100, 200, 400, 600, 1000}`. Training chunk 1 is job `9392814`;
  replacement orchestrator `9393138` will submit chunk 2 and then the 60K eval
  chain. The previous numbers below remain the source of truth
  until that re-run lands; once it does, this section will be rewritten
  against the new data.*
- The seven-root Dysts long-horizon comparison is complete. Dense sparse-latent
  Koopman is best in aggregate at `H5000/H10000` with median best-periodic MSE
  `0.1285 / 0.9778`.
- Block-diagonal sparse-latent Koopman with the higher sparsity penalty is best
  at `H20000/H30000` with `1.9150 / 2.2720`.
- The sparse MLP and non-sparse MLP controls are fully measured:
  sparse MLP `0.1953 / 1.2373 / 3.2524 / 3.6981`; non-sparse MLP
  `0.2474 / 1.4564 / 3.2354 / 3.7893`.
- Block-diagonal MLP controls are competitive at shorter long horizons but do
  not overturn the very-long-horizon winner:
  `0.1501 / 1.1401 / 3.0536 / 3.5891` and
  `0.1945 / 1.2761 / 2.9519 / 3.4785`.
- The older fair `200k` cross-system benchmark remains useful supporting
  context at `H100-H3000`. It is a three-way read rather than a simple
  sparse-versus-dense result: sparse MLP is best at `H100/H1000`
  (`2.947e-4 / 0.0240`), promoted dense sparse-latent Koopman is best at
  `H500` and `H1500-H3000` (`0.0047 / 0.0449 / 0.0627 / 0.0880 / 0.1039`),
  and zero-sparsity improves late-horizon coverage on more systems than it
  wins by median.

Interpretation:
- Sparse-latent Koopman models remain competitive for long-horizon forecasting.
- The dense-to-block-diagonal crossover suggests stronger transition structure
  can matter most at the longest horizons.
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
- Convert the seven-root comparison into one concise table and use the
  all-roots `H30000` best-root phase portraits as visual appendix material.

Primary artifacts:
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
  needs a hard-system stress-test paragraph.
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
- [PAPER_TRACK_STATUS.md](/home/mila/l/lia/skae/docs/PAPER_TRACK_STATUS.md), broad support-audit and label-free clustering entries
- [EXPERIMENTS_ARCHIVE.md](/home/mila/l/lia/skae/docs/EXPERIMENTS_ARCHIVE.md)
