# Allen--Cahn rebuttal repair (2026-07-19)

## Existing evidence and adjudication

The completed four-well Allen--Cahn pilot is not acceptable as forecasting
evidence.  At grid (16\times16), the two-channel state has (d_x=512) and
all Koopman rows used the required overcomplete (d_z=2048).  At the longest
reported autonomous horizon, (H=200) or physical time (4.0), mean field MSE
over three seeds was 1.068 for the dense tanh Conv-KAE, 1.096 for LISTA, and
1.073 for Sparse-MLP.  On the same held-out trajectories, persistence has mean
field MSE 0.053.  At one stored step (physical time 0.02), persistence has MSE
(5.7\times10^{-5}), while the neural rows are near 0.33.  The learned models
therefore do not demonstrate useful forecasting, even though physical time
4.0 is four reaction times and is not merely a short numerical horizon.

The original dense row is nearly dense in measured activations (about 99.6%
active at threshold (10^{-4})) and uses tanh convolutional activations with a
linear latent output.  Its explicit sparsity-loss coefficient is zero, and the
printed nonzero sparsity value is a diagnostic that is multiplied by zero.
The optimizer nevertheless applies (10^{-4}) decoupled weight decay to the
encoder and decoder.  Although this does not create exact activation zeros, the
rebuttal rerun removes it so the no-sparsity control has no shrinkage mechanism
that a reviewer could reasonably contest.  Existing support families collapse
to one family for all three rows, and no trajectory reaches the preregistered
0.7 majority-fraction slice.  The old pilot is therefore a failed experiment,
not positive extension evidence.

## Stage A: physical-regime screen (prediction card)

**Question.** Can a recognizable four-well vector Allen--Cahn system produce a
high-dimensional, globally multistable forecasting task whose trajectories
change enough to make persistence nontrivial and whose long-time basin labels
are unambiguous?

**Frozen factors.** Use a periodic (16\times16) grid, two field channels,
continuum-scaled Laplacian, four wells of radius 1.5, RK4 integration, and
evaluation-only nearest-well labels.  Training and model selection will not
receive labels, a basin count, selected wells, or trajectory assignments.

**Screened factors.** Screen diffusion and stored observation cadence using
dataset-only diagnostics.  The screen may also lengthen physical simulation
time, but it may not change the four-well system after test metrics are viewed.

**Prediction.** Increasing diffusion and using a coarser stored cadence will
produce substantial domain coarsening.  A viable setting should retain all
four final global basins, have median long-label majority fraction at least
0.75, and make persistence materially imperfect at the main physical horizon.

**Decision rule.** Freeze the least aggressive setting satisfying the label
coverage and majority criteria and with persistence final-step MSE at least
0.15 at a horizon of at least four reaction times.  If no screened setting
satisfies this, do not train a model on the benchmark; revise the initial-
condition protocol before any test-set model comparison.

## Stage B: model and optimization screen (prediction card)

**Question.** Does the paper's sparse-support mechanism improve long-horizon
forecasting and basin-support alignment over an uncontestably dense tanh
Koopman autoencoder on the frozen high-dimensional PDE?

**Rows.** Compare (i) persistence, (ii) a dense convolutional Koopman
autoencoder, (iii) a shrinkage sparse-MLP convolutional Koopman autoencoder,
and (iv) a LISTA sign-split convolutional Koopman autoencoder.  Add a direct
convolutional autoregressive predictor before promoting the result to the
paper.  All Koopman rows use (d_z\ge4d_x=2048), matched convolutional trunks,
dense latent transition matrices, and equal optimization budgets.

**Dense-control audit.** Every hidden activation is tanh; the latent output is
linear; no ReLU, GELU, soft-threshold, top-k gate, (L_1) term, sparsity loss,
or activation threshold appears in its forward or training path.  Use zero
weight decay and zero transition regularization.  Record the realized active
density without using it for selection.  Automated assertions must fail the
run if this contract is violated.

**Data and selection.** Increase training-trajectory diversity and use only
translation/reflection/rotation symmetries shared by the periodic PDE.  Tune
learning rate, sparse threshold, rollout-window physical duration, and capacity
on training/validation data.  Select one configuration per method on a fixed
validation objective that includes long-horizon field error and rejects
collapsed representations.  Lock configurations before final test evaluation.

**Primary forecast readout.** Plot normalized field RMSE and skill relative to
persistence against physical time, with individual seeds and paired
uncertainty.  Report reconstruction error separately so autoencoding failure
cannot masquerade as a dynamics result.  Secondary physical metrics are
gradient error, final-well-map IoU, global final-basin accuracy, and strict
finite-rollout rate.

**Primary representation readout.** Fit support families on validation states,
freeze the representatives, and score untouched test states.  Report active
density, family count, (H(B\mid F_{\rm abs})), (H(F_{\rm abs}\mid B)), NMI,
ARI, and purity.  Known basin labels are evaluation-only.

**Prediction.** More data and symmetry augmentation should remove the severe
test reconstruction failure.  LISTA sign-split should expose several stable
support families and outperform the dense row at long physical horizons; the
dense row should remain more than 99% active under the reporting threshold.

**Promotion gate.** Require positive forecast skill over persistence at the
main long horizon, a practically meaningful sparse-vs-dense paired improvement
(target at least 10%), a noncollapsed frozen support-family partition, and the
same directional result on at least five independent seeds.  A paper-facing
claim requires a locked confirmation run and paired bootstrap or randomization
uncertainty.  If the screen misses these gates, tune only on validation or
change the benchmark protocol before regenerating test evidence.

## Stage B validation result and interpretation

At physical time 8, the best dense validation row in the initial grid has mean
field MSE 0.03622 and persistence skill 0.859.  The best sign-split LISTA row
has MSE 0.04003 and skill 0.844, so it is 10.5% worse than dense rather than the
predicted 5--15% better.  The sign-split code is genuinely sparse (active
density 0.499).  On a label-free even/odd split of validation trajectories it
uses 22 transferred late-time support families, has top-family mass 0.181 and
mean within-trajectory modal-family mass 0.728, but its frozen-family coverage
is 0.741, below the preregistered 0.80 guard.  The best shrinkage sparse-MLP
rows are within about 1% of the dense forecast but remain 93--100% active and
collapse to one support family.

These results occur despite useful absolute forecasting: all leading rows beat
persistence by more than 84% at physical time 8.  They therefore validate the
repaired physical task and training pipeline, but they do not validate a
sparse-model advantage.  In particular, the sparse-MLP label must not be used
as evidence of sparse behavior when its realized code is effectively dense.
The result implies that the first optimization screen misses the promotion
gate and that test forecasts must remain locked.  The next step is a second,
frozen validation stage that corrects periodic boundary handling, trains over
longer physical windows, tests data diversity, and includes a non-Koopman
convolutional control.

## Stage C: focused validation repair (prediction card)

**Question.** Does physically correct convolutional boundary handling and a
longer training horizon reveal a reproducible sparse-support benefit, or is the
high-dimensional PDE result limited to strong forecasting and representation
diagnostics without a sparse-vs-dense forecast gain?

**Correction and frozen controls.** All new convolutions use circular padding,
matching the periodic PDE rather than introducing artificial zero-valued
boundaries.  Dense and sparse KAE rows retain the same tanh convolutional
trunks, overcomplete latent dimension 2048, identity initialization of the
latent transition, Adam with zero weight decay, zero transition regularizer,
and equal 2,000-step autoencoder plus 3,500-step dynamics budgets.  The dense
fail-closed no-sparsity audit remains mandatory.  Test fields remain unread by
model evaluation or selection.

**Frozen validation grid.** For dense and sign-split LISTA, cross training-set
sizes 128 and 512 with rollout windows of physical duration 2, 4, and 8 and
transition learning rates (10^{-6}) and (3\times10^{-6}).  The main
sign-split threshold remains (10^{-4}); three larger thresholds are a
support-stability sensitivity.  Also test two-loop signed LISTA with thresholds
0.03, 0.10, and 0.15 and label-free latent penalties 0.003 and 0.01, plus the
same strong-threshold sensitivity for sparse MLP.  Batch size is reduced as
window length grows so each update processes approximately the same number of
state frames.  For each retained checkpoint, evaluate autonomous rollout with
no refresh and with label-free decode/re-encode refresh every 5, 10, or 20
stored steps; the same menu applies to every KAE row.  Selection minimizes
physical-time-8 validation field MSE over this declared rollout menu, and a
sparse row is eligible only if it passes the already frozen label-free
noncollapse/transfer guard.

**Direct control.** Train a circular-padding residual convolutional
autoregressor with tanh activations.  Its output layer is initialized to zero,
so the untrained network is exactly persistence; autonomous rollout repeatedly
applies only its learned update.  It receives no basin labels or counts.  This
control tests whether a generic local neural time-stepper explains the KAE
forecast result without a Koopman lift.

**Prediction and decision rule.** Circular padding and physical-duration-4 or
-8 training windows should reduce long-horizon drift.  A useful sparse result
must both pass the support guard and improve physical-time-8 MSE by at least 10%
relative to the best dense row.  If no sparse configuration passes, do not
evaluate a selectively chosen sparse row on test and do not claim a PDE
forecast advantage.  It remains permissible to lock a label-free
representation candidate and evaluate its basin-support alignment, but that
claim must be separated from forecasting.  Confirmation requires at least five
paired initialization seeds and comparison to the selected direct control.

## Complementary Lorenz--96 confirmation (prediction card)

Existing generated Lorenz--96 evidence is encouraging but not yet a rebuttal
result.  With 128 observed coordinates, a 512-dimensional overcomplete lift,
and three paired data/model seeds, sign-split LISTA has mean NRMSE 0.8832 and
0.9525 at physical times 2.5 and 5.0, versus 0.9131 and 1.0081 for dense MLP
KAE.  The paired trajectory-bootstrap differences favor LISTA at both horizons.
The old dense row does use tanh hidden activations, a linear output, and zero
sparsity coefficient, but its AdamW weight decay is (10^{-6}); moreover, each
model seed also changes the generated dataset.  This is useful provenance, not
the clean independent-initialization confirmation requested here.

**Frozen rerun.** Generate one Lorenz--96 dataset at seed 20260719 with 128
training, 32 validation, and 64 test trajectories, 128 observed coordinates,
5% training-scale observation noise, and stored step 0.05.  Hold that dataset
fixed across ten model-initialization seeds.  Compare the previously selected
sign-split LISTA recipe with a fail-closed dense KAE: tanh encoder and decoder
hidden activations, linear latent output, no ReLU/GELU/soft-threshold module,
zero sparsity coefficient, zero weight decay, Adam rather than AdamW, and no
transition penalty.  Both use (d_z=512=4d_x), dense transitions, equal
3,000-update caps, training horizon 20, and matched test trajectories.  DMD,
truncated DMD, and persistence remain non-neural references.

**Prediction and decision rule.** Sign-split LISTA should retain lower NRMSE
than the exact dense control at physical times 2.5 and 5.0.  Report individual
seed effects and uncertainty that resamples model seeds as the independent
unit; a nested trajectory bootstrap may be shown only as conditional
within-dataset uncertainty.  Promote this as high-dimensional chaotic
forecasting evidence only if the mean effect is favorable at both horizons and
the physical-time-5 seed interval excludes zero.  Lorenz--96 is not
multibasin, so it cannot substitute for the Allen--Cahn basin-support test.

**GPU operating-point audit.** A runtime-only profile on the frozen training
and validation path compared batches 512, 1,024, and 2,048 for 1,000 dense
updates without inspecting test predictions or selecting on validation error.
On the assigned Quadro RTX 8000, their training phases averaged approximately
52%, 90%, and 97% GPU utilization and took 12.86, 11.77, and 20.56 seconds,
respectively.  Batch 1,024 is frozen because it is the smallest setting with
high utilization and the fastest measured setting.  Both learned rows use the
same batch and update cap.

## Independent-seed PDE confirmation (prediction card)

The focused screen uses seed 0 only and is therefore configuration selection,
not confirmation.  After validation selection, freeze one exact-dense KAE, at
most one sparse KAE that passes the label-free support guard, and one direct
convolutional control.  Confirmation uses the same frozen dataset and model
seeds 1--10; seed 0 is excluded because it selected the recipes.  Each seed
may select its checkpoint only by the already declared physical-time-8
validation objective.  Any periodic refresh cadence is fixed globally from
the seed-0 screen rather than selected per confirmation seed.

The direct control is already frozen at 2,500 updates, tanh circular
convolutions, a 20-step training window, batch size 96, learning rate
(3\times10^{-4}), zero weight decay, and the best physical-time-8 validation
checkpoint among 500-step evaluations.  Its confirmation can therefore start
before the KAE screen ends.  Dense and sparse KAE confirmation must wait for
the focused grid and refresh evaluation to lock their configurations.

Report individual model-seed effects and paired seed uncertainty.  Test fields
remain sealed until all recipes and the global refresh cadence are recorded.
The final test pass then evaluates every frozen seed exactly once at physical
times 4, 8, and 12.  Forecasting and representation conclusions are separate:
a sparse forecast advantage requires the preregistered 10% seed-mean gain,
whereas a basin-support claim requires a transferred validation-fitted family
codebook with its coverage and both conditional entropies reported.

For basin-support alignment, the primary collection is each trajectory's final
observed state at physical time 12; five evenly spaced states from physical
times 6--12 are a temporal-stability sensitivity.  Validation representatives
are frozen before test scoring.  A test state transfers only at Jaccard overlap
at least 0.50, and a late-scope trajectory requires at least three of five
states to transfer.  Low-overlap trajectories retain an explicit unknown
family instead of being forced to their nearest representative.  Report both
coverage and metrics conditional on transferred trajectories, plus a
conservative all-trajectory result in which unknown is its own family.

## Locked validation result and confirmation decision

**Concrete result.** The focused seed-0 screen selected the exact-dense row
`dense_n512_l40_klr1e-06` and the sign-split row
`sign_n512_l20_klr1e-06`, both with autonomous rollout and no periodic
decode/re-encode refresh.  Their physical-time-8 validation field MSEs are
0.02974 and 0.03016, respectively, so the sparse row is 1.42% worse rather
than at least 10% better.  Its active-coordinate density is 0.4993 versus
0.9984 for dense.  The sparse row passes the frozen label-free transfer guard:
validation transfer coverage is 0.8219, mean accepted Jaccard overlap is
0.6610, and 18 support families are used.  The independently selected direct
convolutional control is substantially stronger, with seed-0 MSE 0.00710 at
physical time 8 and 0.01031 at time 12; across seeds 1--10 its time-12
validation MSE ranges from 0.00643 to 0.01195 and final-well-map IoU from
0.923 to 0.952.

**Experimental context.** Every Koopman row has a 512-dimensional physical
state and an overcomplete 2,048-dimensional latent.  The dense baseline uses
tanh hidden activations, a linear latent output, Adam, zero weight decay, zero
sparsity and transition penalties, and a full trainable dense transition
matrix.  The sign-split row also has zero explicit sparsity loss; sparsity is
induced only by its declared signed shrinkage encoder.  Validation screening
included refresh cadences 5, 10, and 20, but refresh degraded the leading rows,
so no-refresh is frozen globally.  Seed 0 selected configurations and is
excluded from confirmation inference.

**Interpretation.** The repaired protocol establishes that this physical
system is forecastable and that the old catastrophic KAE result was not caused
by insufficient simulated time alone.  It does not establish a sparse-KAE
forecast advantage: the predeclared promotion gate failed, and the direct
control shows that a generic local time-stepper is far more accurate.  The
only unresolved sparse-KAE claim is representation: whether validation-fitted
sparse support families transfer to untouched test trajectories and align
with the four evaluation-only basin labels without collapsing or fragmenting.

**Project implications.** The paper may use Allen--Cahn as convincing
512-dimensional realistic-physics multibasin evidence only if the independent
test confirms basin--support alignment with adequate coverage and low values
of both conditional entropies.  Forecast plots must show the direct control
and persistence alongside both KAEs, and the text must explicitly state that
sparsity does not improve PDE forecast error under this protocol.

**Next step.** Freeze the two selected KAE recipes and run paired model seeds
1--10 on validation.  If the dense audits and sparse transfer guards remain
valid, evaluate the already frozen direct/dense/sparse ensembles exactly once
on the sealed test split at physical times 4, 8, and 12.  Aggregate over model
seeds with paired exact sign-flip tests and seed bootstraps; use final-state
support transfer as primary and the late-state collection as sensitivity.

**Pre-test representation claim gate.** Normalize
\(H(B\mid F)\) by the test basin entropy and \(H(F\mid B)\) by the transferred
family entropy, retaining unknown as its own family.  On the primary final-state,
all-test slice, sparse support alignment is sufficiently strong only if the
seed means have at least 0.80 trajectory-transfer coverage, normalized
\(H(B\mid F)\leq0.25\), normalized \(H(F\mid B)\leq0.35\), and adjusted Rand
index at least 0.50.  Comparative evidence additionally requires the sparse
mean to improve both normalized conditional entropies and adjusted Rand index
over exact dense, with the paired seed-bootstrap interval for the ARI
advantage excluding zero.  NMI, purity, covered-only, deep-basin, and late-state
results remain declared diagnostics rather than alternative promotion routes.
This rule is recorded before any confirmation model accesses test fields.

**Predeclared displays.** Forecast curves use stored physical time, show every
model seed, and distinguish final-time relative RMSE from time-averaged rollout
relative RMSE; persistence is the unit-error reference.  Support panels put
coverage, one minus each normalized conditional entropy, and ARI on a common
higher-is-better scale with paired seed traces.  The qualitative field display
is fixed to model seed 1 and the first trajectory in the stored test split,
showing truth, direct, dense, and sparse well maps at physical times 0, 4, 8,
and 12.  No trajectory will be selected after viewing test predictions.

## Validation-only support-transfer rescue (prediction card)

The first four completed sign-split confirmation seeds are sufficient to
reject the initial support protocol before test access: seeds 3 and 4 have
label-free even-to-odd validation transfer coverage 0.7625 and 0.7250 and fail
the frozen 0.80 guard, while seeds 1 and 5 pass at 0.8594 and 0.8156.  The
dense rows for these seeds pass every exact-zero-sparsity audit.  All ten
initial seeds will finish so the instability is quantified, but they are now a
validation-only calibration cohort and will not provide confirmatory test
inference.  The test split remains sealed.

**Frozen rescue menu.** Keep the trained sign-split encoder, absolute support
threshold (10^{-3}), late-state scope, even/odd validation split, and all
forecast settings fixed.  Recompute only the label-free family Jaccard rule at
\(\tau\in\{0.35,0.40,0.45,0.50\}\) on initial seeds 1--10.  Select the largest
threshold having at least 9/10 noncollapse-guard passes, seed-mean transfer
coverage at least 0.85, seed-mean effective family count at least 3, and
seed-mean top-family mass at most 0.50.  These generic diversity requirements
do not use or encode the four-basin count.  If no threshold qualifies, abandon
this encoder as a representation candidate and return to architecture tuning;
do not inspect test labels.

**Fresh confirmation.** If a threshold qualifies, train the already frozen
direct, dense, and sparse recipes at fresh paired initialization seeds 11--20.
Promotion to sealed test requires at least 9/10 sparse validation guards plus
the same aggregate coverage, effective-family-count, and top-mass conditions.
All ten fresh seeds then remain in test summaries, including any individual
guard failure.  The original final-state representation claim gate is
unchanged except that its family Jaccard threshold becomes the validation-
selected value.  This rescue card is recorded before evaluating the threshold
menu or training any fresh seed.

Before test access, an evaluation-only validation check will also fit final-
state representatives on even-indexed validation trajectories and score their
transfer to odd-indexed validation trajectories against the stored benchmark
fate labels.  Labels are used only to score this frozen map, never to fit the
encoder, representatives, threshold, or family assignments.  Test promotion
requires the fresh sparse seed means to meet the same absolute alignment
thresholds as the eventual test claim---coverage at least 0.80, normalized
\(H(B\mid F)\leq0.25\), normalized \(H(F\mid B)\leq0.35\), and ARI at least
0.50---and to have higher mean ARI than exact dense.  Failure ends this PDE
representation claim without opening test; it does not trigger another choice
on fresh seeds.

**Concrete calibration result.** The label-free rescue selects
\(\tau=0.45\), the largest qualifying threshold.  Across calibration seeds
1--10 it passes the noncollapse guard 10/10 times, with mean even-to-odd
late-state transfer coverage 0.9175, effective family count 7.47, top-family
mass 0.255, and active density 0.4993.  Thresholds 0.35 and 0.40 also qualify
but are not selected; 0.50 fails with 4/10 passes and mean coverage 0.790.

**Calibration context and interpretation.** Only the family-merging Jaccard
threshold changes.  Encoder weights, absolute activation threshold, training
losses, forecasts, support scope, and all test-facing rules remain fixed.  The
result shows that the original instability came from an overly strict transfer
match rather than support collapse: the selected map preserves more than seven
effective families while no family holds more than roughly one quarter of
accepted states.  It does not yet show basin alignment because no basin label
entered this selection.

**Implication and next step.** Freeze 0.45 for all dense and sparse family
formation, run fresh paired seeds 11--20, and apply the two pre-test validation
gates above.  The canonical selection JSON and row CSV have SHA-256 hashes
`958e3b531b7dd5a24c8aa3bc16ee66918f82a9e19c28d22f595d8a7e010779ea`
and `296d97876bf15c825f1f94283d15eb02ffd3871ccc101f45c16f4ae87bbd6ad2`.

## Lorenz--96 confirmation result

**Concrete result.** On the frozen 128-dimensional Lorenz--96 test set, the
signed sparse KAE has mean NRMSE (0.8987) at physical time 2.5 and (0.9669)
at time 5.0, versus (0.9595) and (1.2889) for the exact-dense tanh KAE.
This is a mean per-seed reduction of 6.30% (95% seed-bootstrap interval
[5.09%, 7.32%]) and 24.87% ([22.98%, 26.51%]); sparse wins all 10 paired model
seeds at both horizons.  Exact two-sided sign-flip (p=0.001953) at each
horizon and Holm adjustment over the two declared long horizons gives
(p=0.003906).  At the same horizons, sparse also improves over persistence by
34.27% and 30.69%, ordinary DMD by 6.90% and 2.30%, and truncated DMD by 6.07%
and 1.91%; every comparison wins 10/10 seeds and has a conditional seed
interval excluding zero.

**Experimental context.** One dataset seed is held fixed across ten model
initializations, with 128/32/64 train/validation/test trajectories, 128
observed coordinates, stored step 0.05, 5% training-scale observation noise,
and an overcomplete 512-dimensional latent.  The dense row has tanh encoder
and decoder hidden activations, a linear latent output, no ReLU, GELU, or
shrinkage module, zero sparsity coefficient, zero weight decay, Adam rather
than AdamW, no soft-block or transition penalty, and a full dense trainable
operator; all ten fail-closed audits pass.  Its empirical active density at
(10^{-4}) is 0.9996, versus 0.4884 for sparse.  Batch 1,024 was frozen from a
runtime-only utilization audit and the full job averaged 81% GPU utilization,
with sparse training phases near 94--96%.

**Interpretation.** This is strong fixed-data evidence that the signed sparse
lift is both materially more stable than a genuine no-sparsity KAE and
competitive with deterministic linear-system baselines through a physical
horizon of 5.0.  The growing dense--sparse gap rules out a claim based only on
a trivially short one-step reconstruction interval.  Because the generated
dataset is fixed, the inference covers model-initialization variability
conditional on that dataset; it does not quantify variability over new
Lorenz--96 datasets.  Lorenz--96 is chaotic but not multibasin, so it cannot
establish basin--support alignment.

**Project implications.** Promote this experiment as the paper's clean
high-dimensional forecasting confirmation.  It directly answers the concern
that the old dense comparison might contain implicit sparsity and provides a
stronger result than the older three-seed packet whose model seeds changed the
data.  Keep the Allen--Cahn test as the separate high-dimensional multibasin
representation question.

**Next step.** Preserve the seed-level rows and audit provenance in the active
paper evidence directory, show NRMSE against physical time with all model
seeds and classical references, and state the fixed-dataset scope explicitly.

## Provenance

The failed pilot is stored under
`/network/scratch/l/lia/skae/allen_cahn_multistable_pde_h200_periodic_50k_20260629`.
Repair code and exploratory runs use the isolated worktree
`/network/scratch/l/lia/skae-rebuttal` on branch
`rebuttal/allen-cahn-highdim`, based on repository checkpoint `c27490d`.
Only compact, adjudicated evidence should be migrated back into the active
paper tree.
