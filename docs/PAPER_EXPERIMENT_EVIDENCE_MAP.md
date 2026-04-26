# Paper Experiment Evidence Map

Date: April 26, 2026

This is the paper-facing map for organizing the NeurIPS experiments section. It
compresses the live experiment record around the evidence chain in the draft:
multibasin Koopman learning motivates sparse supports, sparse supports produce
inspectable chart indicators, and those chart indicators should be useful for
routing prediction. Labels and known basin counts remain evaluation-only
signals on benchmark systems; they are not part of training-time method design
or deployment routing.

Use this file as the first stop for drafting. Use
[EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md) for detailed
experiment logs, [PAPER_TRACK_STATUS.md](/home/mila/l/lia/skae/docs/PAPER_TRACK_STATUS.md)
for high-level claim status and wrap-up priorities, and
[SUPPORT_OBJECT_GLOSSARY.md](/home/mila/l/lia/skae/docs/SUPPORT_OBJECT_GLOSSARY.md)
for exact support-object definitions.

## Recommended figures and tables

The experiments section should use figures and tables in the same order as the
reader's current chain: multibasin problem, supports as chart indicators,
model-produced supports, and support-based routing. The main text should avoid
queue chronology and instead use each display item to close one link in that
chain.

### Main-text display order

1. **Figure 1: Support maps recover basin-local chart structure.**
   Purpose: show that the model produces discrete support objects that align
   spatially with benchmark basins when labels are used only for evaluation.
   Recommended panels: three representative fixed-`17` systems as rows; true
   basin labels, learned exact or family support labels, top-`8` support-family
   labels, support entropy or fragmentation map, and support-versus-basin
   confusion matrix as columns. Use systems with complementary behavior, such
   as one clean polygonal/multiwell case, one gated local-linear case, and one
   transition-rich or boundary-stress case. Caption rule: explicitly state that
   basin labels are evaluation overlays, not training inputs.

2. **Table 1: Fixed-`17` basin-support alignment and in-benchmark forecasting.**
   Purpose: quantify the first link of the chain. Columns should include
   `H100/H500/H1000`, `H(B|S)`, `H(S|B)`, `U_exact`, and freeze or robustness
   ratio if retained. Rows should separate sparse-latent Koopman, sparse MLP
   Koopman, and no-sparsity MLP controls. Keep standard-sampling rows separate
   from hard-init/near-boundary-sampling rows; do not mix them into one causal
   architecture table.

3. **Table 2: Non-oracle support-routed prediction.**
   Purpose: establish that the same support objects can select useful local
   predictors without oracle basin labels. The main table should foreground
   exact top-`8` routing because it is the most deployment-like support object.
   Columns should report routed/global error ratio and win rate on all states
   and deep-basin states. Include the no-sparsity MLP control because it is the
   clearest negative routing comparator.

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
   beyond the controlled low-dimensional multibasin benchmark. Columns should
   be `H5000/H10000/H20000/H30000` median best-periodic MSE. Keep this table
   after alignment and routing so it reads as external forecasting support, not
   as the central interpretability evidence.

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
  whether it is clean/deep-basin, transition-rich, boundary-stress, or
  mechanism/falsification oriented, and which main figure/table uses it.
- **Appendix Table A3: Matched hard-init controls.**
  Summarize the near-boundary-sampling sparse MLP, block-diagonal sparse MLP,
  and no-shrink dense MLP controls. This is the guardrail that prevents a
  LISTA-only claim.
- **Appendix Table A4: Centered-chart, zero-intercept, true-geometry, and
  random-partition diagnostics.**
  Use this to show the calibrated mechanism story: centered local charts are
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
  an interpretable basin-local chart indicator after fully label-free training.

Current result:
- On the fixed `17`-system multibasin benchmark, deep-basin absolute-threshold
  supports are basin-pure for the main sparse-latent models. The strongest
  exact-support alignment row is the sparse-latent Koopman autoencoder with a
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
  basin-aligned chart indicators," not "LISTA alone is responsible." The
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
  control row in the locked final packet, the sparse MLP already has basin-pure
  deep supports and competitive forecasting (`H(B|S)=0.0000`,
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
- Deep inside basins, sparse supports often identify basin-local structure. The
  direct evidence is that the locked final comparison is explicitly on the
  `absolute:0.001` / `deep` support slice and all sparse rows in that table have
  `H(B|S)=0.0000`, meaning a support object is basin-pure on that slice
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:3),
  [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_basin_partition_final_seed10_20260409/final_comparison_pass1/transition_rich_final_comparison.md:12)).
  The same rows have low fragmentation or high dominant-support mass
  (`H(S|B)=0.0543`, `U_exact=0.9923` for the dense sparse-latent finalist;
  `H(S|B)=0.2449`, `U_exact=0.9772` for the standard sparse MLP control), and
  support families saturate (`H(F|B)=0.0000`). The matched hard-init sparse MLP
  controls replicate the basin-purity read under the oversampled regime
  (`H(B|S)=0.0000`, `H(S|B)=0.2068`, `U_exact ~= 0.98`, `H(F|B)=0.0000`)
  ([transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1/transition_rich_final_comparison.md:3),
  [transition_rich_final_comparison.md](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416/final_comparison_pass1/transition_rich_final_comparison.md:10)).
- Exact support identity is not a global invariant. Near separatrices and weak
  coordinates, support families or fixed-size top-`k` supports are more robust
  than a single absolute threshold.
- The no-shrink control is a useful negative/guardrail mainly on functional
  robustness, not on every entropy metric: it can still show basin-pure
  thresholded supports on the selected deep slice, but it is much worse in
  hard-init forecasting and freeze robustness. Therefore the interpretation
  should emphasize induced sparsity plus functional stability, not only one
  entropy column.

Project implication:
- The paper should write this as basin-support alignment, not basin-block
  alignment and not oracle clustering. Basin labels are only used to score
  support purity, fragmentation, and dominant-support mass.
- Standard-sampling and near-boundary-sampling rows should stay separate in
  tables. The hard-init controls support induced sparsity more than a
  LISTA-specific architecture claim.
- Table 1 should therefore have one row family for sparse-latent Koopman, one
  for sparse MLP Koopman, and one for no-shrink / no-sparsity MLP controls, with
  standard-sampling and hard-init rows visibly separated. The main prose should
  say that latent sparsity can produce basin-aligned chart indicators, while the
  strongest exact-support compression belongs to the dense sparse-latent
  finalist and the strongest architecture-isolating hard-init evidence belongs
  to the broader induced-sparsity claim.

Next drafting step:
- Build the main interpretability figure: true basin labels, learned
  support/support-family labels, support entropy, and a support-versus-basin
  confusion matrix on representative systems.

Primary artifacts:
- [FIXED17_LISTA_RESULTS_INDEX.md](/home/mila/l/lia/skae/docs/FIXED17_LISTA_RESULTS_INDEX.md)
- [transition_rich_basin_support_metric_definitions.md](/home/mila/l/lia/skae/docs/transition_rich_basin_support_metric_definitions.md)
- [results/transition_rich_hardinit_mlp_controls_seed10_20260416](/home/mila/l/lia/skae/results/transition_rich_hardinit_mlp_controls_seed10_20260416)
- [EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md), fixed-`17`
  basin-support entries

## 2. Do those same supports select useful local predictors without oracle basin labels?

Paper role:
- This is the functional test of the chart-indicator claim. The first
  experiment asks whether learned supports align with basin labels when labels
  are used only for evaluation. This second experiment asks whether the same
  model-produced support objects do useful dynamical work: can a support
  inferred from the current latent state choose a better local predictor than
  applying one learned global latent transition everywhere?
- In the manuscript logic up to the experiments section, this closes the
  second arrow in the chain
  `sparse latent code -> basin-aligned support object -> support-routed local
  prediction`. Without this result, support alignment could be only a static
  clustering observation. With it, the supports become evidence for a
  label-free routing variable that selects local Koopman-style prediction.

What is being tested:
- Freeze the trained model and infer a support object from the latent state;
  do not use true basin labels, known basin counts, or trajectory-to-basin
  assignments for routing.
- Use the inferred support in two distinct prediction diagnostics:
  `support_gated_k`, where the same learned global transition `K` is applied
  after masking the centered latent input by the current support, and
  `support_local_centered` / `family_local_centered`, where the current support
  or support family selects a post-hoc centered local operator fitted after
  model training.
- Evaluate on disjoint forecast trajectories and compare routed prediction
  against the model's learned global transition. Ratios below `1` and high win
  rates mean the support selected a useful local predictor.
- Foreground fixed-size exact top-`8` supports because they are the most
  deployment-like object: every state gets a route key without choosing an
  activity threshold.

Terminology clarification:
- "Routing" should not be written as if the trained model contains separate
  learned Koopman matrices per support. The base trained Koopman autoencoder
  has one learned global transition `K`.
- `support_gated_k` is support-conditioned masking of the global `K`, not
  switching among independently trained local `K_c` matrices. Algebraically,
  the support selects which centered latent coordinates drive `K`, so it is
  closest to selecting useful columns/subspaces of the learned global
  transition.
- `support_local_centered` and `family_local_centered` are literal local-law
  routing diagnostics. They freeze the trained encoder/decoder, fit centered
  local slopes `A_c` after training, and then use the current model-produced
  support or family as the key that selects which `A_c` to apply.
- These local laws are not used during model training. They are post-hoc
  inference/evaluation objects. In the full non-oracle rollout packet, however,
  they are actually used to produce forecasts in the `support_local_centered`
  and `family_local_centered` modes; they are not merely scored passively.

Centered-chart mechanism packet:
- This is a one-step latent-dynamics diagnostic, not an autonomous forecasting
  study. For each trained checkpoint, generate benchmark trajectories, encode
  each state, and form held-out one-step latent pairs `(z_t, z_{t+1})`.
- Label each transition by the current basin, exact support, or support family.
  Basin labels are included as an evaluation/control partition only, not as a
  training or deployment router.
- Split transitions into fit/test subsets. For each partition class `c` with
  enough transitions, fit a ridge-regularized centered local law
  `z_{t+1} ~= center_c + A_c (z_t - center_c)`.
- Compare held-out one-step latent MSE against the learned global `K`, one
  global centered latent slope, count-matched random partitions, and sometimes
  latent-k-means controls. Also evaluate support-gated versions of the learned
  `K` using support prototypes or block-union masks.
- This packet tests mechanism: whether support/family partitions behave like
  local chart labels for one-step latent dynamics, and whether masking the
  global `K` by support exposes useful local subspaces. It does not test full
  autonomous forecasting.

Full non-oracle rollout packet:
- This is the deployment-facing forecasting diagnostic. For each trained
  checkpoint, use one generated trajectory set to fit support/family centers
  and post-hoc local operators, and a disjoint trajectory set for held-out
  autonomous forecasts.
- Start each forecast from only the initial observation: encode `x_0` to
  `z_0`, then roll forward to horizons `H100`, `H500`, and `H1000`.
- In `global_k`, forecast with the base model rule `z_{t+1}=z_t K`.
- In `support_gated_k`, infer the support from the current predicted latent
  state, then apply the global `K` to the centered masked latent state. If the
  required support center is unavailable, fall back to global `K`.
- In `support_local_centered`, infer the exact support from the current
  predicted latent state, select the post-hoc `A_c`, and apply
  `center_c + A_c(z_t-center_c)`. In `family_local_centered`, first map the
  exact support to a support family and then select the family-level local law.
  Missing routes fall back to global `K`.
- The route is recomputed from the model's own predicted latent state during
  rollout, not from the true future state and not from an oracle basin label.
  This is the packet that tests whether the supports can actually be used for
  inference-time forecasting.

Why this matters:
- The paper's premise is not merely that sparse supports can be inspected after
  training. The premise is that supports can identify which local chart or
  local linear law is relevant for the current state.
- Basin-support alignment alone cannot establish this. A support might be
  basin-pure but dynamically useless, or it might only reproduce an evaluation
  label without improving prediction.
- Non-oracle routing is therefore the necessary bridge from interpretability to
  use: it tests whether the support object produced by the model can replace
  an unavailable oracle basin label when selecting local prediction behavior.

Current result:
- The non-oracle self-routed forecasting packet is complete with `510/510`
  runs, `24,600` rows, and `0` failures.
- The strongest current exact-support result uses fixed-size top-`8` supports
  for the dense sparse-latent Koopman model. At `H1000`, support-gated
  prediction has all-slice median ratio / win rate `0.228 / 0.920` against the
  learned global transition, and support-local centered prediction has
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
  identify useful columns/subspaces of the learned global `K`, and does the
  support work as a non-oracle key for post-hoc centered local laws? These are
  related but not the same claim.
- Be explicit that the main trained model still uses one global `K`. The
  masking and post-hoc local-law modes are inference-time/evaluation variants,
  not training-time supervision or oracle routing.

Next drafting step:
- Add a routing table and a controlled-transfer/refresh figure showing
  state-space path, true basin label for evaluation, inferred support object,
  refresh events, route availability, and post-transfer forecast error.
- TODO: Add or clearly label an inference-time forecasting ablation that uses
  the post-hoc support-local or family-local centered laws as a prediction
  rule. This does not require using local laws during training; the question is
  whether a frozen sparse encoder/decoder plus support-selected post-hoc local
  operators improves held-out forecasting relative to global `K` and
  support-gated global `K`.

Primary artifacts:
- [results/transition_rich_self_routed_forecasting_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420)
- [results/periodic_support_refresh_fixed17_seed0_20260425](/home/mila/l/lia/skae/results/periodic_support_refresh_fixed17_seed0_20260425)
- [controlled_transfer_switching_experiment_20260423.md](/home/mila/l/lia/skae/docs/planning/controlled_transfer_switching_experiment_20260423.md)

## 3. Do sparse-latent Koopman models remain competitive for long-horizon forecasting?

Paper role:
- This is the external forecasting stress test. It shows the sparse-latent
  models are not only interpretable on the controlled multibasin benchmark.

Current result:
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
- This benchmark is supporting evidence, not a basin-support alignment test,
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

- Centered local-chart mechanism: support-conditioned centered local laws beat
  the learned global transition on deep in-basin states, with win rates
  `93.1%` for block-diagonal sparse-latent Koopman, `98.6%` for dense
  sparse-latent Koopman, and `100%` for the dense non-sparse MLP control on
  the deepest slice. This supports the local-chart framing but is not
  LISTA-specific.
- True local geometry: the second-audited true-Jacobian/eigendirection packet
  is mixed. LISTA support families often beat random partitions near
  attractors, but the dense non-sparse MLP can have lower absolute projected
  Jacobian error because its chart is closer to identity. Use this as a
  falsification diagnostic, not a headline claim.
- Controlled transfer: dense sparse-latent exact top-`8` supports switch well
  after a deliberate state-space bridge, while family-level switching is strong
  for multiple roots. This supports support switching but should be written
  separately from a deployment-ready intervention/control claim.
- Negative zero-intercept operator selection: the earlier affine-only local-law
  result failed even with oracle basin fits. Keep it because it justifies the
  centered-chart diagnostic and prevents overclaiming exact operator recovery.
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
  supports, and basin-support alignment on the fixed-`17` systems.
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
