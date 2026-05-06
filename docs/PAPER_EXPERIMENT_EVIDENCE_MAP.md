# Paper Experiment Evidence Map

Date: May 6, 2026

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

May 6 protocol clarification: the controlled multibasin models are trained on
broad boundary-emphasized rollout windows from the generator, not on the deep
slice. Table 1 now places the H100/H1000 forecasting columns first; those
columns use all held-out rollouts, not only deep states. The support-diagnostic
columns use the evaluation-only global deep-state slice, defined by the top
quartile of basin-depth margin over held-out states in each system. The
manuscript now states the rationale explicitly: broad training matches the
label-free deployment problem and preserves boundary/transient coverage, while
the deep slice isolates the cleaner basin-interior question for static
support--basin alignment.

May 6 reviewer-scope add-on note: three requested checks are queued but not
landed. The oracle/local-K control already exists in compact Table 2
partition-control form, and a plain p256 LISTA add-on is queued under
[results/oracle_vs_learned_local_koopman_20260506](/home/mila/l/lia/skae/results/oracle_vs_learned_local_koopman_20260506).
The explicit regime-discovery baseline packet is queued under
[results/regime_discovery_local_koopman_20260506](/home/mila/l/lia/skae/results/regime_discovery_local_koopman_20260506),
covering k-means, diagonal GMM, and spectral clustering on raw state, dense
latent, sparse latent values, and binary supports. The supplemental
out-of-generator packet is queued under
[results/out_of_generator_multistable_p256_lista_20260506](/home/mila/l/lia/skae/results/out_of_generator_multistable_p256_lista_20260506)
for a gene toggle, thermal reactor, modified FitzHugh-Nagumo, and buckled
beam. Until these merges land, they should be described only as pending
reviewer-scope audits.

April 29 live recomputation note: the expanded table-refresh artifacts have
landed and the manuscript tables have been recomputed. The fixed-`17`
five-root forecasting source has `1261` rows; Table 1 interpretability has
`34,047` rows and `0` failures; Table 2 self-routed forecasting has `9,796`
rows and `0` failures; the support-refresh packet has `189,708` rows and `0`
failures; the Dysts packet collected `894/900` requested rows, with the six
missing rows confined to LISTA-BD. The controlled multibasin tables strengthen
the paper. This note is superseded for the current Dysts display by the fully
collected `dt x30` packet and the retained-`10` aggregation that excludes the
six-dimensional `dysts:LorenzCoupled` and Chua-family duplicate
`dysts:MultiChua`: Dense MLP is worst at every horizon, all primary sparse
rows beat Dense in aggregate at every displayed horizon, and the aggregate-best
row is Sparse MLP at `H100`, Sparse MLP-BD at `H500`--`H1000`, and LISTA from
`H1500` through `H5000`.
Treat Dysts as an external forecasting/discretization-sensitivity stress test,
not as support-basin evidence. In the manuscript, LISTA-SB is framed as a
soft-block ablation for the controlled support/routing mechanism and as an
appendix-only Dysts diagnostic.

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
point estimates in that intermediate pass used the old trimmed across-system
aggregate, before the May 3 mean-over-systems correction.
The `[K/17]` counts use within-system Wilcoxon/Holm tests over all seed slots
with censored handling for invalid H-step routed/global comparisons. The
matched row weakens exact-support routing but preserves strong family-local
routing, especially at `H1000`.

May 3 Table 2 statistics/narrative update: the main routing table now
foregrounds `F_top8` family-local routing and moves exact/gated `S_top8`
route forms to appendix diagnostics. The displayed ratio is
`10^{mean_s D_s}`, where `D_s` is the within-system IQM of finite seed-level
`log10(routed/global)` ratios and the outer mean is taken across retained
systems. The companion system-wins column uses the censored per-system
comparison, counting finite routed/global-invalid cases as routed wins. Main
significance stars come from a one-sided exact sign-flip test across
system-level finite log effects, Holm-corrected over the displayed
model--horizon cells. This replaces the previous within-system `[K/17]` count
as the main Table 2 inference.

May 3 layout update: the compact routing table and compact support-refresh
table now appear side by side in the manuscript as subtables 2a and 2b. The
Dysts actual-MSE display is therefore the current manuscript Table 3, although
historical notes may still call that Dysts artifact Table 4.

May 4 manuscript prose/display update: the visible paper todos have been
cleared in `docs/neurips_sparse_koopman_multibasin.tex`. The manuscript now
includes the method overview diagram, a compressed experiments protocol, a
completed related-work section and conclusion, and an appendix
ratio-to-Dense Dysts table that supports the average-case interpretation of
the main actual-MSE Dysts display. This is a prose/display alignment update,
not a new experiment result.

May 5 Methods visual update: use
[fig_methods_support_family_pipeline.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_methods_support_family_pipeline.pdf)
as the main Methods schematic. It replaces the earlier sectorized overview with
a cleaner two-panel white-background diagram: a label-free support route from
\(x_t\) through sparse codes and support-family keys to evaluation/routing,
and a zoomed greedy Jaccard support-family construction panel without basin
labels or basin counts. This is a schematic display artifact, not a new
experiment result.

May 6 training-dynamics appendix note: use
[appfig_training_dynamics_gated_local_seed0.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/appfig_training_dynamics_gated_local_seed0.pdf)
and
[appfig_training_dynamics_dysts_dt30_seed0.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/appfig_training_dynamics_dysts_dt30_seed0.pdf)
only as appendix diagnostics of optimization and checkpoint selection. The
controlled panel plots `gated_local_linear`, seed `0`, for the six controlled
rows. The Dysts panel aggregates seed-`0` traces over the retained `10` Dysts
`dt x30` systems by the median over finite per-system metric values at each
step. Both show validation final error, training objective, \(\rho(K)\), and
the logged sparsity-ratio diagnostic. The controlled validation-selected
checkpoint markers mostly occur late for sparse rows, and \(\rho(K)\) stays
close to the unit circle; the Dysts validation proxy is noisier because
chaotic rollout errors can become very large early in training. Neither figure
should be presented as aggregate benchmark evidence or as a claim that
closest-to-one spectral radius is universally optimal.

May 5 p256 visual-draft update: the Support Barcode Map and Alluvial
Basin-to-Support-to-Family drafts have been regenerated from p256
LISTA-family checkpoints only. Use
[fig_support_barcode_map_p256.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_support_barcode_map_p256.pdf)
as the barcode candidate: p256 dense LISTA on `gated_local_linear`, with the
displayed basin-family prototypes overlaid on one shared `0..255` barcode.
Use
[fig_basin_support_family_alluvial_p256_gated_deep_unbold_headers.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_basin_support_family_alluvial_p256_gated_deep_unbold_headers.pdf)
as the current alluvial candidate: p256 dense LISTA on `gated_local_linear`,
seed `0`, global deep-state slice, `S_abs=10^{-3}`, family Jaccard threshold
`0.5`, `3` represented evaluation basins, `3` support families, `393` exact
supports, and family-dominant basin agreement `1.0`. It keeps the original
alluvial geometry and changes only the column headers to normal weight. The
compact polished and all-state `J=0.32` p256 versions remain
sensitivity/alternate artifacts; earlier p64 drafts are superseded and should
not be used as paper-facing figures.

May 5 Jaccard-threshold sensitivity update: the sweep under
[results/support_family_jaccard_threshold_sweep_20260505/full_retained15_deep_subsetfit](/home/mila/l/lia/skae/results/support_family_jaccard_threshold_sweep_20260505/full_retained15_deep_subsetfit)
confirms that the family Jaccard threshold sets the resolution of the
support-family object. Low thresholds merge distinct basins into too few
families; high thresholds can make \(H(B\mid F)\) vanish by splitting each
basin into many families. Keep the main paper at the fixed `J=0.5`
convention, and use the sweep only as appendix/sensitivity evidence. Do not
make a main narrative claim from high-J zero entropy unless the corresponding
family count is shown.

Whole-slice sensitivity note: at the fixed `J=0.5` threshold, the closest
all-state `F_abs` family-count match is Sparse MLP-BD (`4.217` families versus
`4.20` basins), followed by Sparse MLP (`4.244`). This should be interpreted
as count compression only, because the same rows have higher whole-slice
\(H(B\mid F_{\rm abs})\) than the LISTA-family rows. The main paper should
continue to use the deep-state slice as the cleaner basin-interior alignment
test.

May 5 routed-forecasting-MSE completion note: the experiment plan is
[ROUTED_FORECASTING_MSE_PLAN.md](/home/mila/l/lia/skae/docs/ROUTED_FORECASTING_MSE_PLAN.md).
Two smoke jobs completed with `0` failures (`9466086` controlled multibasin,
`9466087` Dysts), and the full add-on completed raw-row collection:
`9466089`--`9466223` plus merge `9466224` wrote `675` retained-`15`
controlled rows, and `9466225`--`9466314` plus merge `9466315` wrote `450`
retained-`10` Dysts rows. Both merges have `0` failures and complete
model-system-seed coverage. Aggregation job `9467050` then compared routed
`F_top8` family-local centered rollouts with the current same-model
best-periodic Table 1/Table 3 rows using seed IQM within each system followed
by arithmetic means over systems. The result is negative: routed forecasting
wins `0/15` controlled systems and `0/10` Dysts systems for every LISTA-family
model at every evaluated horizon. Do not alter the main evidence order or add
these rows to Table 1/Table 3 as superior forecasting evidence; at most, keep
the result as a falsification/mechanism check showing that useful support
families for local routing do not automatically yield stable autonomous
forecasting rollouts.

May 5 periodic routed-forecasting rerun note: the autonomous routed result did
not settle the user's intended route-refresh hypothesis, so the evaluator was
extended with periodic decode/re-encode via `--reencode_periods`. The rerun is
now complete for periods `5,10,20,30`: compile job `9467340`, smoke jobs
`9467341`/`9467342`, controlled shards `9467343`--`9467477` plus merge
`9467478`, Dysts shards `9467479`--`9467568` plus merge `9467569`, and
aggregation job `9467598` completed with `0` recorded failures. The merged raw
outputs contain `2700` controlled rows and `1800` Dysts rows. Periodic refresh
reduces many autonomous blow-ups and improves route coverage, but it does not
make post-hoc \(F_{\rm top8}\)-family local maps a superior forecasting method:
every aggregate routed/best-periodic ratio remains above `1`, with only
isolated system wins in individual cells. Keep this as a
falsification/mechanism check, not as a main Table 1/Table 3 forecasting row.

May 6 stage-2 fixed-setting expansion result: the `F_top8`, `J=0.40`,
period-`5`, `reroute_each_step` support-family-local \(K_c\) expansion has
now landed enough evidence to decide paper positioning. The 10-seed 50k
controlled multibasin run completed with `0` failures and high route coverage
(`0.9982`), but it is mixed against the same-root best-periodic sparse-LISTA
rollout: routed/best-periodic ratios are
`1.70/0.91/0.84` at `H100/H500/H1000`, with only `4/15`, `5/15`, and `5/15`
system wins. Against the current hard-init Dense MLP Table 1 comparator, the
same controlled row is positive: stage-2/Dense ratios are
`0.0866/0.0513/0.0490`, `14/15` systems have lower per-system seed-IQM MSE at
each horizon, and system-level one-sided Wilcoxon p-values Holm-correct to
`9.16e-4` across the three horizons. The matching 10-seed 50k Dysts run is strongly negative:
`0/10` system wins at every horizon and routed/best-periodic ratios already
`6.9e3` at `H100`. The 100k controlled continuation completed but is worse
than 50k, while Dysts 100k has `97/100` successful rows, three local-map
class-count failures, and unstable available metrics. Table 1 now includes
this as `LISTA + local \(K_c\)`, labeled as a trained local-map forecasting
variant with support diagnostics inherited from the frozen LISTA encoder. It
should be interpreted as Dense-competitive/significant, not as a best-periodic
sparse-LISTA replacement or a Dysts-robust forecasting method.

May 6 Dysts stage-2 re-encode-period sweep queued: the period-`5` Dysts
support-family-local result above is being stress-tested over periods
`{1,2,5,10,20}` by reusing the completed period-`5` rows and queuing the
missing periods `{1,2,10,20}`. Seed-half merges are `9478074` and `9478276`;
combined analysis job `9478278` will write
`results/routed_stage2_local_maps_20260506/combined_best_lista_dysts_j040_50k_period_sweep_labelnone_seed0_9`.
The controlled multibasin result is fairly competitive at `H500/H1000`, but
it is already significantly better than the current hard-init Dense MLP Table
1 comparator. The display plan should therefore allow a controlled-table
stage-2 row if needed, while keeping Dysts robustness pending and avoiding an
external-robustness claim unless the period sweep overturns the Dysts failure.

May 6 calibrated-global stage-2 ablation result: the matched
`stage2_map_mode=global_dense_calibrated` 50k raw-row batches completed for
controlled and Dysts with `0` failures, but this is not a positive calibration
control. Controlled simple means are `H100/H500/H1000 = 16.4/6.86e31/0.125`,
and Dysts simple means are catastrophic from `H100` onward. Combined
aggregation jobs exposed script/schema issues, so raw rows and seed-half
summaries are the current source. The paper-facing interpretation is that
second-stage rollout calibration alone is not a reliable forecasting fix.

Current manuscript layout: pair the alluvial candidate with the seed-level
distribution panels for \(H(B\mid F_{\rm abs})\) and \(|F_{\rm abs}|\) in one
side-by-side support-alignment figure. Omit the wrong-support ratio strip from
the main-text visual display; the functional wrong-support evidence remains in
Table 1 and appendix per-system tables. Keep the active-index codebook as a
separate full-width figure.

May 3 retained-benchmark aggregation update: at the user's direction,
`multiwell_strong_transition` and `claude_checkerboard_potential` are now
excluded from the controlled multibasin benchmark, leaving `15` retained
systems. Main point estimates use seed-IQM within each retained system
followed by an arithmetic mean across retained systems. Tables 1--3 have been
regenerated under this estimator: every sparse-latent row beats Dense MLP on
retained-system mean raw MSE at `H100`, `H500`, and `H1000`; Table 2 routing
denominators are `15` retained systems; and the support-refresh display has
`12` eligible transfer systems. The Dysts `dt x30` table has also been
regenerated after excluding `dysts:LorenzCoupled`, the only six-dimensional
member of the original Dysts shortlist, and `dysts:MultiChua`, to keep a
nonredundant `10`-system three-dimensional shortlist. On the retained systems,
all primary sparse rows beat Dense MLP in aggregate at every displayed
horizon. LISTA passes the system-level Wilcoxon/Holm test at every horizon;
LISTA-BD misses only `H500`, and the two sparse MLP rows miss only `H5000`.
The aggregate-best row is Sparse MLP at `H100`, Sparse MLP-BD at
`H500`--`H1000`, and LISTA from `H1500` through `H5000`.

April 30 matched Table 1/support-refresh completion note: the matched
`d_z=256` Table 1 interpretability diagnostics and support-refresh rows completed with
`0` failures. Interpretability jobs `9412879 -> 9412880` wrote `6,885` rows;
support-refresh jobs `9412881`--`9412883 -> 9412884` wrote `191,400` rows.
Tables 1--3 were recomputed on allocation `9420481` and the manuscript/PDF now
use the matched LISTA-SB row where available. Dysts matching is deferred to the
later full-horizon run.

May 1 dense-K LISTA add-on completion note: the plain dense-K LISTA row is now
complete and folded into Tables 1--3 under
[results/transition_rich_lista_dense_p256_hardinit_table123_20260430](/home/mila/l/lia/skae/results/transition_rich_lista_dense_p256_hardinit_table123_20260430).
The new `lista_dense_signsplit_p256_hardinit_basin_partition` task table has
`255` rows (`17` systems x `15` seeds) and matches the current LISTA Table
1--3 training settings: hard-init sampling, `d_z=256`, `sequence_length=8`,
`200000` optimization steps, sparsity coefficient `0.003`, and a dense Koopman
matrix with no soft-block or block-diagonal K regularizer. Training array
`9423749` completed `255/255` tasks, and collector/evaluation jobs
`9423750`--`9423760` completed with exit `0:0`. The artifacts contain `255`
forecasting rows, `6,885` interpretability rows, `1,980` self-routed rows,
and `191,400` support-refresh rows with `0` failures. The table/figure
builders were rerun on compute allocation `9431949`, and the manuscript
fragments now show the plain LISTA row.

May 1 support-refresh redesign completion note: the support-refresh display has
been rebuilt with period groups rather than a `Fallback` column. Period means the
number of Koopman rollout steps between decode/re-encode support refreshes
after controlled target-basin entry. The missing MLP-control refresh rows
completed under
[results/periodic_support_refresh_mlp_controls_seed0to14_20260430](/home/mila/l/lia/skae/results/periodic_support_refresh_mlp_controls_seed0to14_20260430):
Sparse MLP-BD shards `9423980`--`9423982`, Sparse MLP shards
`9423983`--`9423985`, Dense MLP shards `9423986`--`9423988`, merge `9423991`.
All shards completed with exit `0:0` and shard-level `failure_count=0`.
The first SLURM merge `9423991` ran out of memory and dependent table job
`9423996` was cancelled; clean merge job `9432117` was rerun with `64G`,
completed with exit `0:0`, and wrote `572,654` data rows with `0` failures.
The current support-refresh fragment displays LISTA, LISTA-SB, LISTA-BD, Sparse
MLP-BD, Sparse MLP, and Dense MLP.

May 2 compact control-table rebuild note: the Table 2 partition-control
summary and compact TeX fragment have been rebuilt with the matched
`d_z=256` LISTA-SB row, and the paper-facing compact display now reports
`H100` only. The combined input is
[results/transition_rich_table2_controls_p256_compact_20260502/self_routed_controls/self_routed_forecasting_rows.csv](/home/mila/l/lia/skae/results/transition_rich_table2_controls_p256_compact_20260502/self_routed_controls/self_routed_forecasting_rows.csv);
it keeps the non-LISTA-SB rows from the original control packet and replaces
the p64 LISTA-SB rows with the completed p256 follow-up
(`9429041`--`9429043 -> 9429044`). Final rebuild allocation `9437124`
rewrote
[table2_partition_controls_h100_censored_seed15_summary.csv](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table2_partition_controls_h100_censored_seed15_summary.csv)
and
[table2_partition_controls_compact.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table2_partition_controls_compact.tex).
Use the compact table as matched-dimension appendix/audit evidence. Within
each model, the selectors are ordered Oracle basin labels, Support family,
Latent clusters, then Random matched.

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

May 1 higher-dimensional hard-system correction note: the focused sparse-KAE
forecasting redo for fixed 8-basin CLV, Hopfield `N=16/P=16`, and Kuramoto
identical `N=16` is complete and corrected. The purpose was to revisit the old
hard-system forecasting evidence under the shifted sparse Koopman-autoencoder
narrative with a clean tanh Dense MLP baseline instead of the old ReLU
zero-sparsity control. Training array `9412218`, collector `9412219`, and
comparisons `9412220`--`9412222` completed with exit `0:0` under
[results/hard_system_sparse_kae_redo_p1024_seq8_100k_halflr_sc6em3_tanh_dense_20260429](/home/mila/l/lia/skae/results/hard_system_sparse_kae_redo_p1024_seq8_100k_halflr_sc6em3_tanh_dense_20260429).
The initial collector output under-counted Hopfield because it selected latest
runs per `n_16/seed` wrapper; the fixed collector and allocation `9432839`
refresh now yield the expected `270` rows. It uses `d_z=1024`,
`sequence_length=8`, `100k` steps, half learning rates, `15` seeds, and six
recipes: Dense MLP, Sparse MLP, Sparse MLP-BD, LISTA, LISTA-BD, and LISTA-SB.
The result is negative for the sparse/LISTA recipes: tanh Dense MLP wins all
CLV/Hopfield/Kuramoto system-median comparisons at `H100`, `H500`, and
`H1000`.

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
   The current main-text Figure 1 is the generated composite
   [fig_benchmark_support_dysts_composite.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_benchmark_support_dysts_composite.pdf).
   Top row: `gated_local_linear`, `claude:transition_routes_4`,
   `claude:cal_square_4`, and `claude:cal_high_cross_3` with true
   vector-field streamlines, evaluation basin maps, and LISTA `topk:8`
   support families mapped post hoc to their dominant evaluation basin. The
   polished display uses the concise row label
   `Support-basin alignment`, removes in-panel agreement annotations, and
   keeps the support/basin match-mismatch cue outside the panels with a
   slightly larger legend font. Bottom row:
   Chua, Dadras, Shimizu-Morioka, and Lu-Chen-Cheng H5000 Dysts `dt x30`
   phase portraits from the best seed-`0` primary model in the existing
   all-model manifest, with a slight Dadras zoom. Caption rule: explicitly
   state that basin labels are evaluation overlays, not training inputs, and
   that Dysts is only a forecasting benchmark.
   Candidate-screen note:
   [fig_benchmark_support_grid_agreement_candidates.json](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_benchmark_support_grid_agreement_candidates.json)
   scores the same visual grid-agreement metric for fixed-`17` systems with
   explicit basin maps or catalog attractor-center conventions.
   `claude:cal_high_cross_3` (`0.902`) is now included as the fourth panel
   because it was the strongest unused screen candidate.
   Companion active-index display:
   [support_family_index_codebook_claude_cal_asymmetric_3.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/support_family_codebooks_retained15/support_family_index_codebook_claude_cal_asymmetric_3.pdf)
   shows the actual latent coordinate indices activated by `d_z=256` LISTA-BD
   `topk:8` support-family prototypes on the three-basin Asymmetric wells
   system. Use it when the reader needs to see that the support-family labels
   correspond to concrete coordinate sets, not just colors on a basin map. Row
   labels and tick opacity encode within-basin family coverage; thin tick
   position on the shared `0..255` axis remains the binary active-coordinate
   index. Do not reintroduce the removed printed active-index list column.
   Screening batch:
   [support_family_codebooks_retained15](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/support_family_codebooks_retained15)
   contains one LISTA/LISTA-BD active-index codebook per retained controlled
   multibasin system plus a contact sheet and manifest. Use it for appendix
   selection or for swapping the main companion subfigure; note that the
   Asymmetric wells triplet has been overwritten by the one-panel LISTA-BD
   main-text figure. Do not include all `15` in the main text unless the
   narrative explicitly needs benchmark-wide visual coverage.
   Exploratory \(F_{\rm abs}\) screen:
   [support_family_codebooks_retained15_fabs](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/support_family_codebooks_retained15_fabs)
   repeats the retained-`15` LISTA/LISTA-BD codebooks with support rule
   `absolute:0.001`. Use it only as a diagnostic view of absolute-threshold
   support structure unless the manuscript needs to foreground \(F_{\rm abs}\)
   rather than the cleaner fixed-size \(F_{\rm top8}\) routing object.
   Deep-slice \(F_{\rm abs}\) screen:
   [support_family_codebooks_retained15_fabs_deep](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/support_family_codebooks_retained15_fabs_deep)
   repeats the retained-`15` LISTA/LISTA-BD codebooks on generated observation
   states restricted to the global top-quartile basin-margin deep slice
   (`128 x 128`, eval seed `42`). This is the visual diagnostic closest to the
   main \(F_{\rm abs}\) Table 1 slice, with the caveat that the global criterion
   can leave some systems with only one represented deep basin.
   Additional p256 visual candidates:
   [fig_support_barcode_map_p256.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_support_barcode_map_p256.pdf)
   gives a compact state-space support-family map plus a shared latent
   coordinate barcode for p256 LISTA on `gated_local_linear`, and
   [fig_basin_support_family_alluvial_p256_deep.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_basin_support_family_alluvial_p256_deep.pdf)
   gives a clean deep-slice basin-to-exact-support-to-family flow for p256
   LISTA on `claude:cal_hexagon_6` seed `6`, with exact supports shown as
   unlabeled middle bars and a `6`-basin-to-`6`-family match. The polished
   Image-1-style deep-slice alluvial is
   [fig_basin_support_family_alluvial_p256_gated_deep_unbold_headers.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_basin_support_family_alluvial_p256_gated_deep_unbold_headers.pdf):
   p256 LISTA on `gated_local_linear`, seed `0`, `393` exact supports, and a
   `3`-basin-to-`3`-family match. The all-state sensitivity alluvial is
   [fig_basin_support_family_alluvial_p256_gated_all_j032.pdf](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/fig_basin_support_family_alluvial_p256_gated_all_j032.pdf):
   p256 LISTA on `gated_local_linear`, seed `7`, `S_abs=10^{-3}`, family
   Jaccard threshold `0.32`, `5635` exact supports, and a
   `3`-basin-to-`3`-family match.
   Treat these as alternatives or appendix companions to the current composite
   and active-index codebook, not as p64 carryovers.

2. **Table 1: Retained-`15` forecasting and support-label agreement.**
   Purpose: quantify the first link of the chain. Columns should include
   retained-`15` forecasting at `H100/H1000` first, followed by the current
   support-family alignment and functional-ablation columns: `H(B|S_abs)`,
   `H(B|F_abs)`, wrong-support ratios at `h=1` and `h=20`, and
   arithmetic-mean `|F_abs|`.
   The support-family alignment and wrong-support columns are global
   deep-state-slice diagnostics; the retained-`15` forecasting columns are
   all-held-out-rollout MSEs and should not be described as deep-slice
   forecasting.
   The family count is a non-directional compression diagnostic, not a
   significance-tested loss; the main support-diagnostic figure should show
   the per-seed `|F_abs|` distribution with mean bars for that count panel and
   gray first-to-third-quartile I-bars for distribution spread. Retain
   `|S_abs|` as an appendix active-coordinate diagnostic. Rows should separate
   sparse-latent Koopman, sparse MLP Koopman, and no-sparsity MLP controls.
   The May 5 Jaccard sweep makes the count column mandatory for interpreting
   the entropy column: high thresholds can trivially lower \(H(B\mid F)\) by
   overfragmenting the support families, while low thresholds merge basins.
   The May 6 stage-2 support-family-local row is now included here as
   `LISTA + local \(K_c\)`. It clears the competitiveness/significance versus
   Dense MLP criterion on the retained `15` controlled systems. Its
   caption/prose distinguishes this from the stronger same-root best-periodic
   comparison, where it is mixed, and from Dysts external robustness, where
   period sensitivity is still pending.
   Keep standard-sampling rows separate from hard-init/near-boundary-sampling
   rows; do not mix them into one causal architecture table.
   *2026-05-03 retained-benchmark update.* The table now excludes
   `multiwell_strong_transition` and `claude_checkerboard_potential` and uses
   seed-IQM within retained systems followed by arithmetic means across the
   `15` retained systems. The retained benchmark has evaluation basin-count
   mean/median `4.20/4`, and the global deep slice represents `3.00/3` basins
   on average/median. The `d_z=256` LISTA-SB row reports
   `H(B|F_abs)=0.0442 [11/11]`, wrong-support ratios `2.60e4 [11/11]` at
   `h=1` and `156 [10/11]` at `h=20`, arithmetic-mean `|F_abs|=3.1`, and
   forecasting `H100=0.0387 [15/15]`, `H1000=0.130 [15/15]`. Plain LISTA
   reports `H(B|F_abs)=0.0391 [11/11]`, wrong-support ratios
   `2.19e4 [11/11]` and `110 [10/11]`, `|F_abs|=3.2`, and forecasting
   `H100=0.0407 [15/15]`, `H1000=0.166 [15/15]`.

3. **Table 2: Non-oracle support-routed prediction.**
   Purpose: establish that the same support objects can select useful local
   predictors without oracle basin labels. This table is necessary because
   support-label agreement is only a static membership test; routing tests
   whether the support identifies useful latent coordinates or local laws for
   prediction. The main table should foreground `F_top8` family-local routing:
   exact top-`8` masks are the route primitive, but merged support families are
   the paper-facing deployment object because they absorb brittle exact-mask
   variants without using basin labels.
   Current columns report all-state `F_top8` family-local routed/global ratios
   at `H100` and `H1000`, plus a system-wins count from the censored
   per-system comparison.
   Include the no-sparsity MLP control because it is the clearest negative
   family-local routing comparator. Exact gated/support-local `S_top8` route
   forms belong in appendix diagnostics.
   *2026-04-28 statistics note, superseded by the May 3 system-level rule below.* Use
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
   *Repeatable Table 2 testing rule.* The displayed cell value is
   `10^{mean_s D_s}`, where `D_s` is the within-system IQM of finite
   seed-level `log10(routed/global)` ratios and the outer mean is taken across
   the `15` retained systems. The main significance marker is a one-sided
   exact sign-flip test across the system-level log effects, Holm-corrected
   across the displayed model--horizon cells. The system-wins count uses the
   old seed-slot censoring convention directionally: finite routed with invalid
   global is a routed win, invalid routed with finite global is a loss, and
   both-invalid/missing rows are neutral. The old censored Wilcoxon/Holm count
   is retained only as provenance/appendix diagnostics for route-form
   ablations.
   *2026-05-03 support-family main-table note.* The matched `d_z=256`
   self-routed row remains the manuscript Table 2 LISTA-SB row, but the main
   display no longer asks exact-support routing to carry the paper claim.
   LISTA-family `F_top8` routing clears the corrected system-level test at
   both horizons; Sparse MLP-BD clears at `H100` only; Sparse MLP does not
   clear after correction; Dense MLP remains a negative family-local control.
   Interpret exact support identity as threshold- and dimension-sensitive;
   interpret support-family routing as the robust local-partition result.
   *2026-05-03 layout note.* In the manuscript, this routing display is now
   subtable 2a beside the support-refresh display, which is subtable 2b.
   *2026-05-01 control-table audit.* The same-protocol control pass completed
   under
   [results/transition_rich_table2_controls_20260430/self_routed_controls](/home/mila/l/lia/skae/results/transition_rich_table2_controls_20260430/self_routed_controls)
   with jobs `9425249`--`9425263 -> 9425264`, `12,245` rows, and `0`
   failures. The current H100-only robust summary artifacts are
   [table2_partition_controls_h100_censored_seed15_summary.csv](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table2_partition_controls_h100_censored_seed15_summary.csv)
   and
   [table2_partition_controls_compact.tex](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026/_tables/table2_partition_controls_compact.tex).
   This table supplies the intended control chain: Oracle basin labels ask
   whether local laws help when the benchmark partition is supplied,
   support-family selection asks whether the learned sparse representation
   provides a label-free substitute, and random/cluster partitions test whether
   generic subdivision explains the gain. The compact table now uses the
   matched `d_z=256` LISTA-SB follow-up under
   [results/transition_rich_table2_controls_lista_sb_p256_20260501/self_routed_controls](/home/mila/l/lia/skae/results/transition_rich_table2_controls_lista_sb_p256_20260501/self_routed_controls),
   completed as `9429041`--`9429043 -> 9429044` with `2,475` rows and
   `0` failures.
   *2026-05-01 dense-K LISTA note.* Plain LISTA is now included in the main
   routing table. It has finite-ratio IQMs `0.787/0.775/0.339` for
   gated/support-local/family-local at `H100` and `0.688/0.687/0.009` at
   `H1000`, with family-local clearing `10/17` and `14/17` systems.

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
   *2026-05-01 expanded support-refresh note.* The period-grouped
   support-refresh fragment now
   includes LISTA, LISTA-SB, LISTA-BD, Sparse MLP-BD, Sparse MLP, and Dense
   MLP. All
   displayed rows clear `12/12` transfer-covered systems at both periods. The
   plain LISTA MSE ratios are `0.073 [12/12]` at period `1` and
   `0.019 [12/12]` at period `10`; Sparse MLP-BD and Sparse MLP are also
   strong (`0.055/0.014` and `0.053/0.013`). Dense MLP also clears `12/12`
   with ratios `0.175` and `0.028`. Interpret this as a broad stale-route
   repair effect from periodic re-encoding after moving to a new region, not
   as a LISTA-specific proof that the sparse support is the only useful local
   representation.

5. **Table 3: Dysts long-horizon forecasting.**
   Purpose: answer whether sparse-latent Koopman models remain competitive
   beyond the controlled low-dimensional multibasin benchmark.
   *2026-05-03 paper-facing update.* The main table now uses the fully collected
   `dt x30` sensitivity packet rather than the older tiny-step `H<=60000`
   stress-test packet. It reports
   `H100/H500/H1000/H1500/H2000/H3000/H4000/H5000` on the retained
   `10` three-dimensional Dysts systems, excluding the six-dimensional
   `dysts:LorenzCoupled` system and the Chua-family duplicate `dysts:MultiChua`
   from the originally collected `12`-system shortlist. It uses length-`10`
   training windows, `100k` optimization steps,
   and re-encode periods `{10,25,50,100,150,200}`. Cell point estimate is the
   arithmetic mean across systems of per-system seed-IQM MSEs. The model-vs-Dense inference
   should be marked directly on this actual-MSE table with `*`/`**`: one
   paired log-MSE difference between the candidate and Dense seed-IQM MSE per
   retained system, one-sided Wilcoxon over the `10` systems, Holm-corrected across all
   model-horizon comparisons. The older `[K/N]` within-system reproducibility
   count should stay in diagnostics or appendix discussion rather than the
   main Dysts table. Under
   that system-level test, LISTA is significant versus Dense at every displayed
   horizon, LISTA-BD is significant except at `H500`, and the sparse MLP rows
   are significant through `H4000` but not `H5000`. LISTA-SB does not survive
   all-comparison Holm correction and should remain an appendix-only diagnostic
   in interpretation; the main Dysts figure omits it. Keep the ratio/log-ratio CSVs as
   appendix/audit diagnostics
   with hierarchical bootstrap intervals and system-level/log-ratio SDs;
   avoid raw-MSE SDs as the main dispersion summary because the Dysts errors
   are heavy-tailed and span orders of magnitude.
   Keep this table after alignment and routing so it reads as external
   forecasting support, not as the central interpretability evidence.
   *2026-05-01 clean LISTA-SB diagnostic note.* The current Dysts LISTA-SB
   row was generated before the cleaned hard-vs-soft `K` comparison: it used
   the older two-loop, sign-split soft-block setup. The task builder has now
   been changed so Dysts LISTA-SB matches LISTA-BD's encoder and differs only
   by using dense `K` with a soft off-block penalty instead of hard
   block-diagonal `K`. The one-system Chua smoke, launcher `9430988`, is
   complete with `2/2` rows and full-horizon finite coverage through `H2000`.
   Cleaned LISTA-SB is mixed relative to LISTA-BD on this seed: the
   LISTA-SB/LISTA-BD best-periodic MSE ratios are
   `0.859/0.938/1.191/1.436/1.507` at `H100/H500/H1000/H1500/H2000`. Do not
   promote LISTA-SB into the main Dysts display. That broader diagnostic is
   now complete under
   the paper-facing Dysts protocol: launcher `9432830` generated `12` seed-`0`,
   `100000`-step clean LISTA-SB tasks with the full
   `H100/H500/H1000/H1500/H2000/H3000/H4000/H5000` horizon grid and verified
   `0` bad rows under the fixed LISTA-SB settings (`1` LISTA loop, ReLU final
   operator, dense `K`, soft-block count equal to the documented
   basin/scroll/lobe count). The collector has `12/12` complete rows and full
   median finite coverage. Clean LISTA-SB seed-`0` IQMs are
   `0.000657/0.013247/0.046570/0.132622/0.274048/0.605152/0.991347/1.37441`,
   which are worse than the old seed-`0` LISTA-SB IQMs at every horizon. The
   superseded `50000`-step/`H<=2000` all-system launcher `9432796` was canceled
   before training started. This diagnostic fixes the setup provenance and
   supports keeping LISTA-SB as an appendix diagnostic rather than a main
   Dysts row.
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
   mixed: dense-transition LISTA is best by the prior trimmed aggregate at every horizon
   (`0.198` at `H5000`, `4.93` at `H60000`) but clears at most `2/12` systems;
   LISTA-BD has catastrophic finite errors and should not be framed as a Dysts
   win.

6. **Figure 3, optional main or appendix: Dysts long-horizon visual packet.**
   Purpose: show representative long-horizon rollouts and avoid a purely
   numeric forecasting section. The paper-facing aggregate trend should use the
   `dt x30` log-scale mean-MSE plot; the seed-`0` `H1000`--`H5000` all-model
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
  Use the current-roster per-basin rerun, not the older completed
  `interpretability_per_basin_deep_pass1` packets, to repeat the Table 1
  support/freeze diagnostics under a per-basin top-quartile deep slice. The
  older packet completed for the earlier root roster, but it lacks the current
  plain `p256` LISTA row and uses the older `p64` LISTA-SB row. The current
  rerun was submitted as jobs `9477837`--`9477845` and writes
  `interpretability_per_basin_deep_current_table1_pass0/` outputs in the three
  current Table 1 source packets. As of `2026-05-06 01:52 EDT`, shard jobs are
  still running with partial `25`--`30`-run summaries per root and `0`
  failures; wait for the merge outputs before drafting this table. This
  appendix should be framed as coverage robustness for the wrong-support
  ablation, not as a replacement for the global deep-slice Table 1 numbers
  unless the paper deliberately changes the support diagnostic estimand to
  per-basin relative-depth coverage.
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
  experiments narrative. Add the corrected `d_z=1024` tanh-dense-baseline
  higher-dimensional redo here as a dense-baseline hard-system stress test
  rather than as a main evidence pillar.

## 1. Do sparse supports align with basins when labels are used only for evaluation?

Paper role:
- This is the first main result. It tests whether the learned support object is
  an interpretable indicator of basin membership after fully label-free
  training.
- It is deliberately a static test. It asks whether the support says where the
  state is, not whether the support already proves which latent coordinates or
  local operator should advance the state.

Current result:
- On the retained `15`-system multibasin benchmark, support families on states
  far from basin boundaries have low conditional basin entropy for every
  sparse-latent row: `H(B|F_abs)=0.0391` for LISTA, `0.0456` for LISTA-BD, and
  `0.0442` for LISTA-SB, versus `0.765` for the dense no-shrink MLP baseline.
- The same retained table shows that wrong-support interventions are strongly
  functional for sparse models: wrong-support/base ratios at `h=1` are
  `2.10e4`--`7.08e4` for sparse rows versus `20.0` for Dense MLP, and the
  effect persists at `h=20`.
- On the same retained-`15` table, every sparse-latent row has lower
  arithmetic-mean raw MSE than Dense MLP at `H100`, `H500`, and `H1000`.
  The displayed `H100/H1000` values are `0.0407/0.166` for LISTA,
  `0.0411/0.135` for LISTA-BD, `0.0387/0.130` for LISTA-SB, and `0.830/2.93`
  for Dense MLP. The main retained-benchmark horizon display uses
  fixed-system, log-relative seed-bootstrap `95%` bands, matching the Dysts
  horizon display.
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
- Foreground fixed-size top-`8` support families because every state first
  gets a threshold-free route key and nearby exact keys are then merged without
  basin labels into a more robust deployment-facing selector.

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
- The safest deployment-facing support object is currently the fixed-size
  top-`8` support family, not an absolute-threshold exact mask or a brittle
  exact top-`8` identity.
- The periodic-refresh claim should be narrowed: exact-support evidence is
  strongest for dense sparse-latent top-`8` supports after basin entry; support
  families support a broader version.

Project implication:
- The experiments section should foreground `F_top8` family-local routing.
  Exact top-`8` route forms should stay as diagnostics showing why family
  aggregation is the robust support object.
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
- Queued: add a direct long-horizon router comparison between \(F_{\rm abs}\)
  and \(F_{\rm top8}\). The matching \(F_{\rm abs}\) stage-2 replay has been
  submitted with `support_definition=absolute:0.001`, `J=0.40`, period `5`,
  `reroute_each_step`, `50000` steps, one GPU per worker on `long`, and
  `DEVICE=cuda`, using the same retained systems, seeds, fitting/evaluation
  split, route-population threshold, fallback rule, and held-out rollout
  protocol as the existing \(F_{\rm top8}\) fixed-setting result. Report
  routed/global MSE, routed/best-periodic MSE where applicable,
  route coverage/fallback rate, occupied-family counts, transition counts per
  family, underpopulated routes, and invalid/blow-up counts. This is required
  before the draft can claim empirical superiority of `F_top8` over `F_abs` for
  deployment routing.
- TODO: Add or clearly label an inference-time forecasting ablation that uses
  the post-hoc centered support-local or family-local laws as a prediction
  rule. This does not require using local laws during training; the question is
  whether a frozen sparse encoder/decoder plus support-selected post-hoc local
  operators improves held-out forecasting relative to the global transition and
  support-gated global transition.
- Queued reviewer add-ons for this section: the plain p256 oracle-vs-learned
  local-\(K\) comparison (`9478597`, `9478599`, `9478601 -> 9478603`) will
  make the learned support-family versus oracle basin gap explicit for the
  current root, and the retried regime-discovery packet
  (`9478633`--`9478647 -> 9478648`) will test whether standard
  unsupervised partitions over raw state, dense latent, sparse latent values,
  or binary supports match the learned support-family local-\(K\) route. Keep
  both out of the evidence map's result claims until the merge artifacts land.
  Compile validation `9478630` and runtime smoke `9478651` passed after fixing
  mixed-type route-label metric encoding.

Primary artifacts:
- [results/transition_rich_self_routed_forecasting_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_20260420)
- [results/periodic_support_refresh_fixed17_seed0_20260425](/home/mila/l/lia/skae/results/periodic_support_refresh_fixed17_seed0_20260425)
- [controlled_transfer_switching_experiment_20260423.md](/home/mila/l/lia/skae/docs/planning/controlled_transfer_switching_experiment_20260423.md)

## 3. Do sparse-latent Koopman models remain competitive for long-horizon forecasting?

Paper role:
- This is the external forecasting stress test. It shows the sparse-latent
  models are not only interpretable on the controlled multibasin benchmark.
- A supplemental out-of-generator multistability packet is now queued as a
  separate generality check rather than as part of the fixed retained
  controlled benchmark. It covers `claude:toggle_switch_3gene`,
  `claude:bistable_reactor`, `claude:fitzhugh_nagumo_3eq`, and
  `claude:buckled_beam` with the current p256 LISTA recipe. Use it only after
  the dependent collect, interpretability, oracle/local-K, and regime-baseline
  jobs land.

Current result:
- *2026-04-29 refresh:* the `seq_len=10`, up-to-`15`-seed, `12`-system
  `H{5K..60K}` Dysts rerun has been collected under
  [results/dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428/collect](/home/mila/l/lia/skae/results/dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428/collect).
  The collector has `894/900` requested rows. The six missing or invalid rows
  are all LISTA-BD (`dysts:DequanLi` seed `3` and `dysts:LorenzCoupled` seeds
  `1,2,4,8,9`), and within-system paired tests use common completed seeds.
- Dense-transition LISTA had the best old trimmed aggregate at every reported
  horizon: `0.198`, `1.02`, `2.88`, `3.81`, `4.30`, `4.65`, and `4.93` for
  `H5000` through `H60000`. However, the within-system Holm-corrected counts
  are weak (`2/12`, `2/12`, `0/12`, `0/12`, `2/12`, `0/12`, `0/12`), so this
  should be described as aggregate competitiveness rather than broad
  systemwise dominance.
- LISTA-BD is not a positive Dysts result in this rerun. Its prior trimmed aggregate is
  dominated by catastrophic finite errors (`7.9e29` at every displayed
  horizon) and it clears `0/12` systems against Dense MLP. Sparse MLP and
  Sparse MLP-BD stay near the Dense MLP baseline and clear at most `1/12`
  systems at the longest horizons.
- *2026-05-03 aggregation-corrected `dt x30` sensitivity:* the full `12`-system coarser-step packet
  completed under
  [results/dysts_dt30_basinblock_p256_seq10_100k_20260430/long_horizon_eval/collect](/home/mila/l/lia/skae/results/dysts_dt30_basinblock_p256_seq10_100k_20260430/long_horizon_eval/collect)
  with `1080/1080` rows, `0` pending/invalid rows, and median full-horizon
  finite coverage of `1` for every root at every reported horizon. The
  paper-facing rebuild now excludes `dysts:LorenzCoupled`, the only
  six-dimensional member of the original shortlist, and `dysts:MultiChua`, the
  Chua-family duplicate. It reports arithmetic means across per-system
  seed-IQMs on the retained `10` three-dimensional systems: at `H5000`, LISTA
  is `0.689`, LISTA-BD is `0.741`, Sparse MLP-BD is `0.760`, Sparse MLP is
  `0.765`, and Dense MLP is `1.08`. The appendix diagnostic LISTA-SB value is
  `0.997`, better than Dense in aggregate but not significant after
  all-comparison correction. The aggregate-best row is Sparse MLP at `H100`,
  Sparse MLP-BD at `H500`--`H1000`, and LISTA from `H1500` through `H5000`.
- The within-system Wilcoxon/Holm read against Dense MLP is positive but not
  universal and remains a diagnostic rather than the main inference. On the
  retained `10` systems, LISTA improves `10/10` systems at every horizon,
  LISTA-BD improves `9/10`--`10/10`, Sparse MLP improves `9/10`--`10/10`,
  and Sparse MLP-BD improves `9/10`--`10/10`. The aggregate Wilcoxon/Holm test
  over retained systems clears LISTA at every horizon, LISTA-BD except `H500`,
  and both sparse MLP rows through `H4000`; LISTA-SB stays appendix-only for
  Dysts.
  The paper-facing raw-MSE trend uses log scale because the displayed IQM MSEs
  span about `4.8e3x`; linear-scale and side-by-side scale-check plots are
  retained as diagnostics. Trend bands are fixed-system seed-bootstrap `95%`
  intervals after system-wise log-relative normalization and are anchored to
  the displayed arithmetic mean, so they quantify typical finite-seed
  uncertainty within an average system rather than variation across the
  retained `10` systems or raw-scale sensitivity to one high-MSE resample. The
  `_raw_seed_ci` plots retain the original raw-MSE fixed-system interval,
  `_system_ci` plots resample fixed per-system seed-IQMs across systems, while
  `_log_seed_bootstrap` plots resample seeds after a `log10` MSE transform. The table and
  plots are under
  [docs/figures/neurips_paper_2026](/home/mila/l/lia/skae/docs/figures/neurips_paper_2026),
  and phase portraits for all `12` originally collected systems at `H1000`--`H5000` are under
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
  shorter-horizon all-system aggregate table, but do not let it reorder the main
  experiments section.

Next drafting step:
- Keep the Dysts table concise and explicitly secondary. If a Dysts visual is
  included, use the `dt x30` log-scale mean-MSE trend as the main view and keep
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
  MLP-better. The corrected `d_z=1024` tanh Dense MLP redo closes the loop in
  the same direction: Dense MLP wins all CLV/Hopfield/Kuramoto system-median
  comparisons at `H100`, `H500`, and `H1000`. This belongs in limitations or
  appendix unless the main paper needs a hard-system stress-test paragraph.
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
