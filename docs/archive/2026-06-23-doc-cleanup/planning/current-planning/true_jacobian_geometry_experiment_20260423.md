# True Jacobian Geometry Experiment

Date: April 23, 2026

## Audit Status

The first seed-`0` fixed-`17` output under
`results/true_jacobian_geometry_fixed17_seed0_20260423_cached` is treated as
**procedure-suspect**. The first corrected output under
`results/true_jacobian_geometry_fixed17_seed0_20260423_corrected` fixed the
projection and centering bugs below, but a second audit found one remaining
accounting flaw: a support/family class was evaluated only at its dominant
recovered fixed point. If the same support object occurred near multiple
attractors, non-dominant local neighborhoods were omitted. The current
paper-facing packet is therefore the second corrected rerun under
`results/true_jacobian_geometry_fixed17_seed0_20260424_reaudit`.

The two projection/centering mistakes in the original evaluator were:

1. The fitted latent operator uses the repository's row-vector convention
   `(z_t - c) @ A`, but the state-space projection used
   `J_decode A J_encode`. With autograd Jacobians in column-vector convention,
   the correct projection is `J_decode A^T J_encode`.
2. The local operator was fit around the empirical mean of the selected
   near-attractor samples and then projected at the recovered fixed point.
   That does not estimate the derivative at the fixed point. The corrected
   evaluator fits the centered operator around `E(x*)`, where `x*` is the
   recovered fixed point, and reports the sample-mean-to-anchor distance as a
   diagnostic.

The second audit fix now emits one row per `(partition class, recovered fixed
point, radius)` whenever that class has local radius coverage, and records
`class_total_count`, `class_near_radius_count`,
`class_fixed_point_fraction`, and `near_radius_fixed_point_fraction`. This is
more conservative for shared support/family objects because it no longer hides
their behavior near non-dominant attractors.

The second corrected rerun completed as SLURM job `9347929` with `49/49` runs,
`198,302` rows, `114,419` ok rows, and `0` failures.

Second corrected result:

- LISTA family partitions still often beat random count-matched partitions
  near attractors, but the margin is weaker after class-attractor accounting.
  At radius `0.15`, blockdiag LISTA family relative-Frobenius error is
  `0.1169` versus `0.1979` random for `absolute:0.001`, `0.1205` versus
  `0.1872` for `relative:0.1`, and `0.1114` versus `0.1955` for `topk:8`.
- Blockdiag family results are mixed at larger radii. At radius `0.6`,
  `absolute:0.001` remains better than random (`0.1296` vs `0.1420`), but
  `relative:0.1` and `topk:8` are worse than random (`0.1389` vs `0.1351`,
  `0.1512` vs `0.1338`).
- Dense LISTA `topk:8` family rows beat random across the radius sweep:
  `0.1328 / 0.1268 / 0.1304` versus
  `0.1541 / 0.1404 / 0.1394`.
- Dense LISTA exact `topk:8` supports beat random across the radius sweep:
  `0.1410 / 0.1329 / 0.1512` versus
  `0.1881 / 0.1591 / 0.1558`, though ok-row coverage remains sparse
  (`76 / 109 / 173` ok rows across radii).
- The zero-sparsity MLP remains a strong caution. Its `topk:8` family rows
  also beat random at radii `0.15` and `0.3` (`0.0732` vs `0.1191`,
  `0.0711` vs `0.0978`) and are roughly tied at `0.6` (`0.0931` vs
  `0.0915`), while exact `topk:8` support rows remain worse than random
  (`0.0847` vs `0.0552`, `0.0764` vs `0.0393`, `0.0692` vs `0.0497`).

Interpretation:

This is not evidence that LISTA exact supports recover true Jacobians or true
eigendirections. The safe use is narrower: support and family objects can
select non-random local slopes in the learned chart near attractors, but the
effect is chart-dependent and not LISTA-specific at the family level. Dense
LISTA exact `topk:8` support remains the strongest support-level positive, but
this branch should stay a secondary falsification diagnostic rather than a
headline result.

## Purpose

This first-pass branch tests the reviewer-facing mechanism claim:

`basin -> sparse support -> local linear law`

The existing fixed-`17` packets already support basin-support alignment and
self-routed forecasting. The remaining vulnerability is that the learned
support-conditioned laws have not been compared to true local geometry near
known attractors or fixed points. This protocol adds a conservative, falsifiable
geometry diagnostic on existing checkpoints. It is not a training run.

## Hypothesis

Deep near an attractor, a basin-aligned support object should select a learned
centered latent operator whose encoder/decoder local projection agrees better
with the true state-space one-step Jacobian at that attractor than a
count-matched random support-family control, provided the encoder/decoder are
locally valid charts at the attractor.

This is deliberately weaker than raw latent matrix equality. It allows latent
basis changes, sparse support conventions, and decoder geometry. The comparison
is made in state space after local projection. Because fitted operators use
row-vector convention, the projected derivative is:

`J_decode(z*) A_support^T J_encode(x*)` versus `J_true_step(x*)`.

This is a local hyperbolic-fixed-point diagnostic. It is not expected to be
valid far from the attractor, for limit cycles, for nonsmooth basin boundaries,
or when the learned encoder/decoder pair is not locally invertible enough to
act like a chart. For that reason the evaluator now sweeps attractor radii and
reports chart-validity diagnostics rather than treating a single neighborhood
as definitive.

## Falsifiable Outcomes

Positive outcome:

- On systems with reliable recovered fixed points, the real support/family
  rows have lower relative Frobenius error and closer eigendirection/spectrum
  metrics than random count-matched rows and are competitive with the
  attractor-local baseline, especially under `absolute:0.001` and `topk:8`
  near attractors.
- The effect is strongest for support objects already known to be basin-aligned
  and weaker or absent for fragmented support definitions.
- The effect strengthens as the neighborhood radius shrinks, unless sample
  count becomes too small.

Negative outcome:

- Real support/family rows are no better than random controls after projecting
  to state space.
- Support/family rows only look good at large radii but fail in tighter
  attractor neighborhoods, which would argue against a true local-geometry
  interpretation.
- The local chart diagnostics fail: high anchor reconstruction error, a
  rank-deficient encoder/decoder Jacobian, or large
  `||J_decode J_encode - I||` makes the projected-Jacobian comparison
  uninterpretable even if forecast metrics are good.
- The projected operator spectra are systematically unrelated to true local
  spectra even when support alignment is high.
- Most systems with claimed attractors fail fixed-point residual checks, which
  means this is not an appropriate geometry test for those systems.

Ambiguous outcome:

- Fixed-point recovery succeeds, but encoder/decoder local Jacobians are
  unavailable because the model path is nondifferentiable at the anchor or the
  recovered point lies off the learned manifold.
- Complex eigendirections dominate; spectrum can still be compared, but
  one-dimensional eigendirection orientation is marked `N/A`.
- Repeated or nearly repeated eigenvalues make individual eigendirection
  matching unstable; these rows are retained for spectral-radius and
  Frobenius diagnostics but eigendirection claims are suppressed.

## Systems Included And Excluded

Included candidates:

- The fixed-`17` transition-rich shortlist discovered from the supplied
  `forecasting_rows.csv` files.
- Native systems with `env.points`, `basin_label`, or `dynamics`.
- Claude catalog systems with `system.dynamics` and recoverable attractor
  candidates from `wells`, `_wells`, `centers`, or `well_centers`.
- Systems whose candidate attractors can be refined by rollout and pass the
  configured step-residual gate.

Excluded by evaluator skip rows:

- Systems without any recoverable candidate fixed point or attractor center.
- Systems whose refined candidate points have residual above
  `--fixed_point_residual_tol`.
- Systems where the true one-step Jacobian cannot be differentiated.
- Rows where the local support/family partition has too few transitions to fit
  a centered operator.
- Rows where encoder/decoder Jacobian projection fails.

This evaluator may use basin labels and recovered basin counts for benchmark
evaluation only. It must not be interpreted as a training-time method design
that assumes known basin labels or known basin count.

## Confounders

- A support can align with basin identity but not with the local slope.
- A centered latent operator can forecast well because of chart offsets rather
  than true local Jacobian agreement.
- Learned latent coordinates can be arbitrarily transformed, so raw latent
  matrix equality is not a valid success criterion.
- The supplied well centers are not always true fixed points when rotation,
  transfer corridors, or confinement are present.
- Non-smooth support selection and ReLU/LISTA thresholds can make local
  encoder Jacobians unstable exactly at sparse boundaries.
- Some transition-rich systems may have limit-cycle or channel behavior rather
  than reliable point attractors.

## Controls

- Same checkpoint/model for all support definitions and controls.
- Same generated trajectory batch per run spec.
- Same basin/depth restriction near recovered attractor points.
- Attractor-radius sweep: default `0.25,0.5,0.75`, because local
  linearization theory supports only a neighborhood claim.
- Support definitions: `absolute:0.001`, `topk:8`, and optionally
  `relative:0.1`.
- Partition kinds: `attractor`, `basin`, `family`, `support`.
- `attractor` is a non-support baseline that fits one local operator per
  recovered fixed-point neighborhood using all covered transitions in that
  neighborhood. It tests whether the learned chart can recover local geometry
  at all before asking whether supports recover it.
- Random count-matched controls made by permuting partition labels on the same
  trajectory batch before applying the same near-attractor filtering. This
  preserves global label counts but does not guarantee identical per-attractor
  local counts, so transition-count fields must be inspected.
- Explicit `N/A` fields and skip reasons instead of silently dropping systems.

## Metrics

Primary state-space metrics:

- `state_fro_rel_error`: `||A_projected - J_true||_F / ||J_true||_F`.
- `eigval_mean_abs_error`: matched eigenvalue absolute error.
- `spectral_radius_abs_error`: absolute difference of spectral radii.
- `real_eigendirection_abs_cos_mean`: mean absolute cosine between matched
  real eigendirections when dimensionally meaningful.
- `true_min_pairwise_eig_gap` and `learned_min_pairwise_eig_gap`: diagnostics
  used to suppress unstable eigendirection interpretation near repeated
  eigenvalues.

Diagnostic metrics:

- `fixed_point_step_residual`: fixed-point residual after refinement.
- `continuous_residual`: `||f(x*)||` when continuous dynamics are exposed.
- `latent_anchor_distance`: distance between fitted latent chart center and
  `encode(x*)`.
- `anchor_reconstruction_error`: `||D(E(x*)) - x*||`.
- `chart_identity_fro_rel_error`: relative Frobenius error of
  `J_decode(z*) J_encode(x*)` against the state-space identity.
- `encoder_jacobian_rank`, `decoder_jacobian_rank`, and condition-number
  diagnostics for local chart degeneracy.
- `transition_count`: number of near-attractor transitions used to fit the
  centered local operator.
- `support_size_mean`: mean support size for the partition prototype.
- `projection_status` and `skip_reason`: explicit status accounting.

## Interpretation

Treat positive results as support for the mechanism only if real
support/family rows beat random rows on the same system/checkpoint, behave
sensibly across the radius sweep, and have acceptable chart-validity
diagnostics. Do not claim that latent operators equal true Jacobians. The safe
claim is that support-conditioned latent local laws, when mapped through a
locally valid learned chart, recover state-space local geometry better than
count-matched random support objects.

Treat negative results as evidence that support alignment alone does not imply
the local-linear-law part of the story. This would not invalidate
basin-support alignment or non-oracle forecasting results, but it would weaken
the phrase "local linear law" unless reframed more carefully.

## Exact Commands

Smoke run under an interactive compute allocation:

```bash
salloc --mem=8G -c 4 --partition=long
ROWS_CSVS=/path/to/forecasting_rows.csv \
OUT_DIR=runs/true_jacobian_geometry_smoke \
ROOT_LABELS_CSV=lista_dense_softblock_signsplit_p64_hardinit_basin_partition \
SYSTEMS_CSV=gated_local_linear \
SEEDS_CSV=0 \
SMOKE=1 \
scripts/run_transition_rich_true_jacobian_geometry.sh
```

Full evaluator submission:

```bash
ROWS_CSVS=/path/to/forecasting_rows.csv \
OUT_DIR=runs/true_jacobian_geometry_fixed17 \
ROOT_LABELS_CSV=lista_dense_softblock_signsplit_p64_hardinit_basin_partition,lista_blockdiag_signsplit_hardinit_basin_partition,mlp_zero_sparse_basin_partition_control \
QUEUE_MANIFEST_JSON=runs/true_jacobian_geometry_fixed17/queue_manifest.json \
scripts/queue_transition_rich_true_jacobian_geometry.sh
```

Direct evaluator arguments used by the run script:

```bash
uv run python tools/evaluate_transition_rich_true_jacobian_geometry.py \
  --rows_csvs "${ROWS_CSVS}" \
  --output_dir "${OUT_DIR}" \
  --root_labels "${ROOT_LABELS_CSV}" \
  --systems "${SYSTEMS_CSV}" \
  --seeds "${SEEDS_CSV}" \
  --support_definitions absolute:0.001,topk:8,relative:0.1 \
  --partition_kinds attractor,basin,family,support \
  --attractor_radii 0.25,0.5,0.75 \
  --num_trajectories 128 \
  --trajectory_length 128
```

Expected artifacts:

- `true_jacobian_geometry_rows.csv`
- `true_jacobian_geometry_summary.md`
- `manifest.json`
- `failures.json`
- `progress.json`

## First-Pass Limitations

- Fixed-point recovery is conservative and may skip systems that require
  dedicated root finding.
- This branch should be reported only on systems with reliable fixed points.
  Skipped systems are not failures of the support mechanism; they are outside
  the diagnostic's validity conditions.
- The evaluator compares one-step map Jacobians by default. Continuous
  dynamics Jacobians are logged when available but are not the primary
  comparison to learned discrete-time operators.
- Eigenspace comparisons are marked `N/A` when complex eigenvectors or repeated
  eigenvalues make one-dimensional orientation misleading.
- If the attractor-local baseline itself fails after chart projection, the
  right interpretation is that this checkpoint/chart is not suitable for a
  true-geometry claim, not that support conditioning specifically failed.
