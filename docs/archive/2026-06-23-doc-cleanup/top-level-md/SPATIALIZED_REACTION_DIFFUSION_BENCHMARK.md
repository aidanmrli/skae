# Spatialized Reaction-Diffusion Benchmark

## Current Status

Worker A owns the PDE path for the appendix spatialized multibasin
reaction-diffusion benchmark. The benchmark now has two implementation paths:

1. The original flattened one-seed smoke path, which generated the first
   feasibility results below.
2. A new convolutional Koopman path for the paper-facing high-dimensional
   benchmark. It keeps the same label-free training contract, uses a
   convolutional encoder/decoder over two-channel fields, supports
   `conv_lista`, `conv_dense`, and `conv_sparse_mlp` variants, and is driven by
   a task-table SLURM array launcher.

One-seed job `9530539` completed successfully. This is a pipeline validation
result, not a publishable benchmark result yet. One-seed tuning and the
same-seed dense control are complete. The May 25 overcomplete interactive
screen was promising, but the first three-system/five-seed expansion is mixed:
dense forecasts better at short and mid horizons, while LISTA mainly provides
more compact supports. The tanh-matched sparse tuning pilot completed and
confirms the tradeoff: much better compact support alignment than dense, but no
decisive sparse forecasting win. Treat this benchmark as an appendix
stress-test candidate unless targeted sparse tuning recovers forecasting
quality.

Latent-dimension rule: all future spatialized PDE runs must use an
overcomplete Koopman lift with `d_z >= 4 * d_x`, where
`d_x = 2 * grid_size^2` for the current two-channel fields. Thus grid `16`
has `d_x=512` and requires `d_z>=2048`; grid `32` has `d_x=2048` and requires
`d_z>=8192`. Earlier `z_dim=128` screens are retained only as invalid
calibration/debugging history and must not be used as paper evidence.

May 25 overcomplete screen: interactive GPU runs under
`/network/scratch/l/lia/skae/spatial_rd_overcomplete_interactive_20260525`
used `grid16`, `d_x=512`, `d_z=2048`, and matched dense controls. The strongest
current setting is `conv_lista`, `alpha=0.2`, `sparsity_weight=0.05`, evaluated
with `support_threshold=0.2`, `family_jaccard=0.2`, and a basin-dominant field
slice `majority_fraction>=0.7`. On that slice, LISTA improves
`H(B|F),H(F|B)` over dense on `cal_square_4` (`0.000,0.231` versus
`0.231,0.549`) and improves `H(B|F)` with equal `H(F|B)` on
`transition_routes_4` (`0.000,0.277` versus `0.382,0.277`). LISTA uses far
smaller supports (`78.4` and `106.9` active coordinates out of `2048`) than
dense (`681.6` and `610.7`). Forecasting is competitive but mixed: LISTA wins
H12 on `cal_square_4` and H4/H8/H12 on `transition_routes_4`; it is mildly
worse than dense at H4/H8 on `cal_square_4`. Interpretation: the
overcomplete PDE path now has a plausible paper-facing signal, but the all-test
slice still fragments near mixed/separatrix fields, so scale-up should report
dominant-slice and all-test metrics separately.

May 25 expansion queue: the first overcomplete statistical expansion is
complete and summarized. Parent jobs `9647692`, `9647693`, and `9647695`
submitted training arrays `9647700`, `9647701`, and `9647702` for selected
LISTA, dense, and sparse-MLP. The arrays cover `cal_high_cross_3`,
`var_l_shape_5`, and `cal_pentagon_5`, seeds `0`--`4`, with `grid16`,
`d_x=512`, `d_z=2048`, `20,000` steps, checkpointing every `1000` steps, and
resume from `last.pt`/`checkpoint.pt`. Runner resources are `long`, `gpu:1`,
`16G`, `4` CPUs, and `02:50:00`. All training arrays completed, and dependent
support-sweep parent `9647703` submitted support array `9648659`, which
completed all `45/45` tasks with exit `0:0`; all `45` support-sweep JSON files
are present under
`/network/scratch/l/lia/skae/spatial_rd_overcomplete_expand_support_sweep_20260525`.
Summary CSVs are under
`results/spatial_rd_overcomplete_expand_20260525/summary`.

Concrete result: dense is the best short/mid-horizon forecaster on the added
systems. Mean H4/H8/H12 field MSE is `0.4278/0.4374/0.5339` for dense,
`0.4842/0.4817/0.5364` for LISTA, and `0.4777/0.4798/0.5476` for sparse-MLP
under the time-averaged rollout metric. Endpoint-only H12 `final_field_mse`
is more favorable to LISTA (`0.7641`) than dense (`0.8746`) or sparse-MLP
(`0.8121`), but the packet is still short-horizon because `trajectory_length=16`
allows ground-truth forecast scoring only up to H16. At the fixed support
diagnostic (`support_threshold=0.2`, `Jaccard=0.2`,
`majority_fraction>=0.7`), dense has `H(B|F)=0.063`, `H(F|B)=0.671`, and
support size `78.5`; LISTA has `0.074`, `0.623`, and support size `19.3`; and
sparse-MLP has `0.034`, `0.637`, and support size `41.4`. Interpretation:
LISTA produces compact supports and modestly less fragmentation than dense,
but does not win the purity entropy or forecasting metrics. Next step is a
small targeted sparse-tuning pass plus a long-trajectory forecast packet, not
a broad scale-up.

May 26 long-horizon pilot: parent `9656822` completed and submitted training
array `9656825` plus dependent support parent `9656826`. Because H256/H512
cannot be evaluated from `trajectory_length=128` fields, this queue stores
`trajectory_length=512` and `label_extra_observations=512`, while the trainer
uses `train_observation_limit=128` and `sequence_length=8` so training windows
come only from the first 128 observation intervals. The pilot covers
`cal_square_4` and `transition_routes_4`, seeds `0,1`, and
`conv_lista`/`conv_dense`/`conv_sparse_mlp`, with `grid16`, `d_x=512`,
`d_z=2048`, `num_steps=30000`, and
`eval_horizons=1,4,8,16,32,64,128,256,512`. The training/evaluation array
completed all `12/12` tasks with exit `0:0`, but the original `conv_dense`
rows are superseded because the PDE convolutional dense path used GELU hidden
blocks instead of tanh hidden activations. The code has been patched, old
support parent `9656826` was canceled, and corrected dense-only parent
`9659093`, training/evaluation array `9659165`, and support-sweep array
`9659514` completed with exit `0:0`. Corrected tanh dense is the strongest
forecaster through H256, with H8/H16/H32/H64/H128/H256 field MSE
`0.242/0.302/0.917/1.441/1.819/2.126`, versus LISTA
`0.695/0.773/1.208/1.623/2.015/2.750` and sparse-MLP
`0.668/0.778/1.260/1.693/2.010/2.942` on finite rows. H512 MSE is non-finite
for every variant. Sparse rows retain compact support alignment at the fixed
diagnostic: deep-slice `H(B|F)`/support size are corrected dense `0.678/151.0`,
LISTA `0.111/7.6`, and sparse-MLP `0.043/11.7`. The current long-horizon
conclusion is a forecast/support tradeoff: useful for diagnosing stability and
support formation, but not a sparse forecasting win.

May 26 tanh-matched sparse tuning pilot: model-capacity audit confirmed that
the corrected dense advantage is not due to larger width, depth, latent
dimension, decoder, or Koopman `K`; all rows use the same `hidden_channels=32`,
`num_blocks=2`, `d_z=2048`, dense `K`, and data/training protocol. The code now
supports a `conv_activation` override so LISTA and sparse-MLP can use tanh
blocks like the corrected dense control. Pytest job `9661520` passed (`10`
tests), queue parent `9661532` submitted training/evaluation array `9661535`
with `48` sparse tuning tasks and support parent `9661536`. The grid is
`lista_alpha=0.02,0.05,0.1` x `sparsity_coeff=0,0.01` for
`conv_lista,conv_sparse_mlp` on `cal_square_4,transition_routes_4`, seeds
`0,1`, still at `grid16`, `d_x=512`, `d_z=2048`, `trajectory_length=512`, and
`train_observation_limit=128`. Current status: training/evaluation array
`9661535` completed `43/48` rows; tasks `12,18,25,37,43` failed only because
they timed out waiting for shared dataset locks, and support parent `9661536`
was canceled. Rescue array `9662516` completed the missing rows with exit
`0:0`, bringing the pilot to `48/48` checkpoints and evaluations. Replacement
support parent `9662517` completed and submitted support array `9662576`
(`48` tasks, `0-47%4`), and support array `9662576` completed all `48/48`
tasks with exit `0:0`.

Concrete result: corrected tanh dense is still best at H1/H4/H8/H16/H32
(`0.289/0.260/0.242/0.302/0.917` field MSE), while the best tanh sparse rows
are `0.307/0.276/0.259/0.412/0.973`. Tanh sparse/LISTA is only modestly
better at H64/H128 (`1.409/1.761` versus dense `1.441/1.819`), H256 has
partial non-finite sparse coverage, and H512 has no finite field-MSE rows.
Support alignment is much stronger than dense. At `threshold=0.05`,
`Jaccard=0.4`, LISTA `alpha=0.05`, `sparsity=0.01` gives
`H(B|F)=0.171`, `H(F|B)=0.241`, `4.75` families, but large support size
`399`. At the more compact `threshold=0.3`, `Jaccard=0.4`, LISTA
`alpha=0.05`, `sparsity=0` gives `H(B|F)=0.283`, `H(F|B)=0.241`, `4.25`
families, support size `17.5`; corrected dense at the same threshold/Jaccard
has `0.658/0.200`, `2.75` families, support size `57.7`. Interpretation:
GELU should not be the paper-facing sparse conv default; run a ReLU
`generic_sparse`-style convolutional sparse path next and keep tanh sparse as
an activation ablation.

A tiny convolutional smoke completed on May 21 and validated imports,
generation, training, checkpointing, evaluation, and the task-table array path
on compute nodes. Direct CPU smoke job `9615803` completed with exit `0:0`.
Task-table launcher `9615822` completed with exit `0:0` and submitted array
`9615825`; all `3/3` array tasks completed with exit `0:0` for `conv_lista`,
`conv_dense`, and `conv_sparse_mlp`.

A small controlled convolutional pilot also completed on May 21. Launcher
`9615832` submitted array `9615833`; all `6/6` child tasks completed with exit
`0:0` for two systems (`cal_square_4`, `transition_routes_4`), seed `0`, and
the three convolutional variants. This validates the paper-facing mechanics at
larger-than-smoke scale, but support-family reads are still degenerate and the
pilot is not paper evidence.

A follow-up support-tuning loop completed on May 21 and found a clean
non-degenerate support setting. The selected setting is `z_dim=128`,
`hidden_channels=32`, `lista_num_loops=3`, `lista_alpha=0.03`,
`sparsity_weight=0.05`, evaluated with `support_threshold=0.3` and
`family_jaccard=0.8`. On two systems and seeds `0,1`, LISTA reaches mean NMI
`0.751` and mean purity `0.938`; sparse-MLP is slightly higher on mean NMI
(`0.776`) and dense is also non-degenerate (`0.708`). This establishes clean
supports under a tuned scale-dependent diagnostic, but not a LISTA-specific
win.

A May 22 re-evaluation used the paper's actual support objects. With fixed
\(S_{\rm abs}=\{|z_i|>10^{-3}\}\) and \(F_{\rm abs}\) Jaccard `0.5`,
\(F_{\rm abs}\) collapses to one family for all selected PDE checkpoints. The
exact \(S_{\rm abs}\) object separates LISTA and sparse-MLP by basin on the
tiny held-out slices, but the supports are dense and fragmented rather than
compact (`~118/128` active coordinates; `95/97` exact supports on all states
for LISTA/sparse-MLP). This corrects the paper-facing interpretation: the
current PDE packet validates benchmark construction and exposes latent-scale
issues, but it should not be promoted as evidence that paper-defined
\(F_{\rm abs}\) discovers compact basin-support families.

## May 21 Convolutional Benchmark Implementation

Concrete implementation changes:

- Added a paper-facing convolutional Koopman model in
  `skae/benchmarks/spatialized_conv_koopman.py`. The public API remains
  flat-field compatible (`encode`, `decode`, `rollout_observation_discrete`),
  but internally reshapes fields to `[batch, channel, x, y]`.
- Added `tools/train_spatialized_reaction_diffusion_conv.py` for
  convolutional LISTA, dense, and sparse-MLP controlled variants.
- Extended dataset generation with a separate label-settling horizon,
  continuum-vs-graph Laplacian scaling metadata, observed-final labels, initial
  area fractions, and invalid/clipped-value audit counts.
- Extended evaluation with same-horizon modal basin consistency,
  final-fate consistency, per-pixel basin-map accuracy/IoU, majority-fraction
  error, Fourier-band MSE, support-size summaries, and frozen validation
  support-family assignment.
- Added `tools/build_spatialized_reaction_diffusion_tasks.py`,
  `scripts/run_spatialized_reaction_diffusion_array.sh`, and
  `scripts/queue_spatialized_reaction_diffusion_benchmark.sh` for multi-system,
  multi-seed task-table experiments.
- Added support-sweep tooling and queue-script support for reproducible
  LISTA-alpha/sparsity/support-threshold/Jaccard sweeps. Array tasks now lock
  shared dataset generation to avoid concurrent writes.
- Added `tests/test_spatialized_reaction_diffusion.py` for helper/model/task
  builder shape and construction checks.

Context:

The changes address the main reviewer-facing risks in the flattened smoke:
high-dimensional fields should be modeled spatially, labels should correspond
to a post-observation fate horizon rather than only the last stored training
state, and evaluation should distinguish same-horizon basin-map accuracy from
early prediction of the final fate label.

Interpretation:

This is an implementation milestone plus a useful support-scale diagnostic, not
a final scientific result. The convolutional path now passes compute-node smoke
checks, a two-system pilot, and a two-seed tuned matched-control read. The
tuned setting gives non-degenerate support-family alignment, but the
manuscript \(F_{\rm abs}\) object collapses at the fixed `1e-3` support rule,
and dense and sparse-MLP controls are competitive.

Project implications:

The benchmark is now valid as runnable infrastructure for a NeurIPS-facing
high-dimensional protocol, but the paper-object support claim is not solved.
Current evidence supports benchmark validity and a diagnostic latent-scale
story, not compact \(F_{\rm abs}\) basin discovery or a unique LISTA advantage.

Next steps:

1. Do not scale the tuned threshold diagnostic as a paper claim without a
   paper-object replacement plan.
2. Either justify a scale-normalized PDE support rule as sensitivity, or test
   \(C_{\rm stab}\) on PDE support trajectories.
3. Keep dense and sparse-MLP controls in any future display because both are
   competitive under tuned support thresholds.

## May 22 Paper-Object Re-Evaluation

Concrete result:

- Scope: the 12 selected matched-control checkpoints from the May 21 tuning
  pass (`cal_square_4`, `transition_routes_4`; seeds `0,1`;
  `conv_lista`, `conv_sparse_mlp`, `conv_dense`).
- Output root:
  [results/spatial_rd_paper_support_eval_20260522](/home/mila/l/lia/skae/results/spatial_rd_paper_support_eval_20260522).
- Focused tests passed on compute allocation `9623724`: `7 passed in 13.51s`.
- The re-evaluation wrote per-run JSON/log files plus
  [paper_support_rows.csv](/home/mila/l/lia/skae/results/spatial_rd_paper_support_eval_20260522/paper_support_rows.csv)
  and
  [paper_support_summary.json](/home/mila/l/lia/skae/results/spatial_rd_paper_support_eval_20260522/paper_support_summary.json).
- On `deep_final_test_states`, mean \(F_{\rm abs}\) family count is `1.00` for
  all three variants; mean `H(B|F_abs)=1.057`, purity `0.469`, and NMI `0.0`.
- On `all_test_states`, mean \(F_{\rm abs}\) family count is also `1.00` for
  all three variants; mean `H(B|F_abs)=1.131`, purity `0.438`, and NMI `0.0`.
- Exact \(S_{\rm abs}\) gives `H(B|S_abs)=0.000` for LISTA and sparse-MLP on
  all reported collections, but with heavy fragmentation and dense supports:
  all-state exact support counts are `95.0` for LISTA and `97.0` for
  sparse-MLP, with mean support sizes `117.5/128` and `117.6/128`.
- Dense has almost full exact supports (`~127.6/128`) and worse exact-support
  basin entropy (`0.506` on deep-final states, `0.789` on all states).

Context:

The May 21 result used a tuned support threshold (`0.3`) and Jaccard (`0.8`) to
diagnose whether the convolutional path could produce non-degenerate support
reads. The May 22 result uses the manuscript definitions: \(S_{\rm abs}\) with
strict threshold `1e-3` and \(F_{\rm abs}\) with fixed Jaccard `0.5`.

Interpretation:

The tuned diagnostic and paper-object read answer different questions. The
tuned diagnostic shows that the latent contains basin-separable information
under a scale-adjusted threshold. The paper-object read shows that the current
conv PDE checkpoints do not produce compact \(F_{\rm abs}\) families under the
manuscript rule; nearly all coordinates are active, so Jaccard merging collapses
families.

Project implications:

The current PDE packet should remain outside the main NeurIPS evidence chain.
It is useful for benchmark construction, scale diagnostics, and future
support-object development, but not for the current \(F_{\rm abs}\)
basin-support-alignment claim.

Next steps:

1. Add a scale-normalized support rule only if it can be justified before
   seeing labels and reported as sensitivity rather than a replacement for
   \(F_{\rm abs}\).
2. Evaluate \(C_{\rm stab}\) on PDE support trajectories if the goal is a
   basin-of-attraction object rather than static support-family alignment.
3. Treat any future scale-up as benchmark-development work until the support
   object is fixed.

Support-tuning and matched-control result:

Concrete result:

- Diagnostic finding: the original conv pilot had dense-support collapse, not
  zero-support collapse. At `support_threshold=1e-4`, LISTA support size was
  about `64/64`, so Jaccard merging produced one family.
- Existing-checkpoint threshold sweep: on the pilot LISTA checkpoint,
  increasing the evaluation threshold already produced non-degenerate reads
  (`cal_square_4`, `tau=0.03`, `J=0.9`: NMI `0.821`, purity `1.0`;
  `transition_routes_4`, `tau=0.1`, `J=0.9`: NMI `0.632`, purity `0.875`).
- Training sweep: `9616021 -> 9616023` varied `lista_alpha` and
  `sparsity_weight`; `9616085` swept support thresholds/Jaccards for the
  `z_dim=128` two-system screen.
- Selected setting: `z_dim=128`, `hidden_channels=32`, `num_blocks=2`,
  `lista_num_loops=3`, `lista_alpha=0.03`, `sparsity_weight=0.05`,
  `support_threshold=0.3`, `family_jaccard=0.8`.
- Matched sparse-control run, seed `0`: launcher `9616098`, array `9616099`,
  all `4/4` child tasks exit `0:0`.
- Matched dense-control run, seed `0`: launcher `9616103`, array `9616104`,
  all `2/2` child tasks exit `0:0`.
- Matched sparse-control run, seed `1`: launcher `9616117`, array `9616119`,
  all `4/4` child tasks exit `0:0`.
- Matched dense-control run, seed `1`: launcher `9616118`, array `9616120`,
  all `2/2` child tasks exit `0:0`.
- Across two systems and two seeds, mean all-test NMI/purity/H4 field MSE:
  LISTA `0.751/0.938/0.970`, sparse-MLP `0.776/0.938/0.977`, dense
  `0.708/0.875/0.962`.
- LISTA per-system mean NMI/purity:
  `cal_square_4` `0.737/0.875`; `transition_routes_4` `0.766/1.000`.

Context:

The tuning used no basin labels in training. Labels enter only through the
evaluation metrics and threshold scans. The final matched-control read fixes
the support threshold from the LISTA tuning pass and applies it to all three
model families.

Interpretation:

The benchmark now has clean support families under a spatial convolutional
model. The result is not LISTA-specific: sparse-MLP is slightly stronger on
mean NMI, and dense is also non-degenerate, though weaker on transition-routes.
This means the tuned benchmark is suitable for studying basin-support
alignment, but not yet for claiming a unique advantage of the LISTA encoder.

Project implications:

The PDE benchmark has moved from implementation validation to a viable
high-dimensional benchmark-development result. It can be scaled, but the paper
claim should be framed as a benchmark/protocol and controlled comparison unless
larger sweeps reveal a clearer sparse-LISTA advantage.

Next steps:

1. Run the selected setting on the full five-system suite with seeds `0,1,2`.
2. Add a threshold-sensitivity appendix plot/table so reviewers can see that
   clean supports are not an artifact of a single threshold.
3. If dense remains competitive, frame the PDE benchmark as evidence that the
   spatialized construction exposes basin-support structure, not as a primary
   SKAE-vs-dense win.

Controlled convolutional pilot result:

Concrete result:

- Launcher: `9615832`, exit `0:0`.
- Array: `9615833`, all `6/6` tasks exit `0:0`.
- Output root:
  `/network/scratch/l/lia/skae/spatial_rd_conv_pilot2sys_seed0_20260521`.
- Scope: `cal_square_4` and `transition_routes_4`, seed `0`,
  `16x16` two-channel fields, `24/8/8` train/validation/test trajectories,
  sequence length `4`, `50` training steps, variants `conv_lista`,
  `conv_dense`, and `conv_sparse_mlp`.
- Dataset audit: both systems produced shape `[40, 13, 16, 16, 2]` with
  invalid values `0` and clipped values `0`.
- Dataset label counts:
  `cal_square_4` `{0: 11, 1: 10, 2: 11, 3: 8}`;
  `transition_routes_4` `{0: 10, 1: 14, 2: 9, 3: 7}`.
- Best validation MSE:
  `cal_square_4`: `conv_lista 0.8990`, `conv_sparse_mlp 0.8991`,
  `conv_dense 0.8997`;
  `transition_routes_4`: `conv_lista 1.6977`, `conv_sparse_mlp 1.7039`,
  `conv_dense 1.7039`.
- Evaluation at `H=4`:
  `cal_square_4` field MSE is `0.8751` for `conv_lista`, `0.8772` for
  `conv_sparse_mlp`, and `0.8780` for `conv_dense`; same-horizon modal basin
  consistency is `0.75` for all three.
  `transition_routes_4` field MSE is `1.6752` for `conv_lista`, `1.6775` for
  `conv_sparse_mlp`, and `1.6770` for `conv_dense`; same-horizon modal basin
  consistency is `0.75` for `conv_lista` and `0.50` for the two controls.
- Support-family read: all six rows have `1` validation representative, NMI
  `0.0`, and purity `0.5`.

Context:

This pilot uses the label-settling generator and the convolutional model path.
Training consumes only field sequences. Basin labels, basin maps, and attractor
metadata are used only by the evaluator.

Interpretation:

The pilot confirms that the dataset construction is numerically stable on the
tested systems and that the convolutional trainer/evaluator can run through a
task-table SLURM array. It does not yet show the desired basin-support
alignment: every variant collapses to one validation support family under the
current short training/tiny-latent settings.

Project implications:

The benchmark is now credible as a constructed high-dimensional multibasin
testbed, but the current pilot supports only an infrastructure claim. A paper
claim needs a non-degenerate support-family read and enough seeds/systems to
separate LISTA from dense and sparse-MLP controls.

Next steps:

1. Retain the current generator/evaluator protocol.
2. Tune the convolutional LISTA setting first; do not scale the degenerate
   support read.
3. Once supports are non-degenerate, rerun matched dense and sparse-MLP
   controls on the same task table.

Smoke result:

- Direct runtime smoke: job `9615803`, exit `0:0`, node `cn-h004`, elapsed
  `34s`.
- Direct smoke dataset:
  `runs/spatialized_reaction_diffusion/conv_smoke_maincpu_short_20260521/cal_square_4_seed0_grid8/dataset.pt`,
  shape `[12, 7, 8, 8, 2]`, label counts `{0: 4, 1: 3, 2: 2, 3: 3}`,
  invalid values `0`, clipped values `0`.
- Direct smoke training:
  `runs/spatialized_reaction_diffusion/conv_smoke_maincpu_short_20260521/cal_square_4_seed0_grid8/conv_lista/training_summary.json`,
  status `completed`, best validation MSE `0.8237`.
- Direct smoke evaluation:
  `runs/spatialized_reaction_diffusion/conv_smoke_maincpu_short_20260521/cal_square_4_seed0_grid8/evaluation.json`,
  status `completed`.
- Task-table smoke: launcher attempts `9615810`/`9615816`/`9615818` exposed
  that CPU child jobs were still inheriting GPU GRES. The queue script was
  patched to distinguish unset `RUNNER_GRES` from explicitly empty
  `RUNNER_GRES=`, and to use `run_spatialized_reaction_diffusion_array_cpu.sh`
  for CPU arrays.
- Successful task-table smoke: launcher `9615822`, exit `0:0`; array `9615825`,
  all `3/3` tasks exit `0:0`.
- Successful array outputs:
  `/network/scratch/l/lia/skae/spatial_rd_conv_array_smoke_20260521_rerun3`.
- Array training summaries: `conv_lista`, `conv_dense`, and `conv_sparse_mlp`
  all status `completed`; best validation MSEs are `0.8237`, `0.8240`, and
  `0.8241`, respectively.
- Array evaluation at `H=2`: field MSEs are `1.0938`, `1.0940`, and `1.0940`
  for `conv_lista`, `conv_dense`, and `conv_sparse_mlp`; all three collapse to
  one validation support family in this intentionally tiny smoke, so NMI is
  `0.0` and purity is `0.5`.

## First One-Seed Result

Concrete result:

- Job: `9530539`
- Output: `runs/spatialized_reaction_diffusion/cal_square_4_seed0_grid32_smoke/evaluation.json`
- Dataset: `72` trajectories, shape `[72, 25, 32, 32, 2]`
- LISTA checkpoint: step `100`
- Forecast field MSE: `1.040` at `H=1`, `1.211` at `H=4`, `1.319` at `H=8`, `1.373` at `H=12`
- Final-basin consistency: `0.0` at all reported horizons
- Support families: `42` validation representatives; test purity `0.583`, NMI `0.313`, ARI `0.011`
- Final-majority diagnostic: no trajectories satisfy `majority_fraction >= 0.9`; the generated fields remain mixed-domain (`majority_fraction_mean=0.502`)

Context:

The run validates that the generator, flattened LISTA training path, checkpoint
save/load, field/gradient forecast metrics, and validation-frozen support-family
alignment all execute on a compute node without using labels during training.

Interpretation:

The scientific signal is not yet acceptable. The latent collapsed to near-zero
under the current sparse settings (`sparsity_ratio=1.0` late in training), and
forecasting did not preserve the final basin. Treat this as a working one-seed
pipeline and a tuning target, not as evidence for the paper.

Project implications:

The next PDE step should remain one-seed tuning, not scale-up. Priorities are
lower LISTA shrinkage/sparsity, an all-held-out support read that does not
require a deep-basin filter, and then a same-seed dense KAE control once the
LISTA path gives non-degenerate supports.

## One-Seed Tuning Result

Concrete result:

- Jobs: `9530782`, `9530783`, `9530784`, plus Jaccard sweep `9530799`
- Best current setting: `LISTA_ALPHA=0.001`, `SPARSITY_COEFF=0`, `NUM_STEPS=1000`
- Best checkpoint step: `700`
- Forecast field MSE: `0.740` at `H=1`, `0.891` at `H=4`, `0.985` at `H=8`, `1.032` at `H=12`
- Final-basin consistency at `H=12`: `0.917`
- Sparse supports no longer collapse: late-training `sparsity_ratio=0.775`
- Best compressed support-family read: support threshold `1e-4`, Jaccard `0.7`, `7` validation representatives, `6` test families, purity `0.917`, NMI `0.709`, ARI `0.502`
- Exact-support read at Jaccard `1.0`: purity `1.0`, but `116` validation representatives and lower ARI (`0.154`), so the merged family object is the better paper-facing diagnostic.
- Final-majority diagnostic remains mixed: no test trajectories satisfy `majority_fraction >= 0.7` or `0.9`; only `4` test trajectories appear under a relaxed `majority_fraction >= 0.5` read.

Context:

The tuning sweep changed only one-seed LISTA shrinkage/sparsity and
validation-time support-family merging. No basin labels, basin counts, or
selected initial centers were used by training.

Interpretation:

The one-seed PDE path now has a working LISTA result with non-degenerate
supports and meaningful all-held-out basin-support alignment. The mixed-domain
final fields should be interpreted as part of the benchmark rather than as a
failed primary slice; majority fraction remains useful for diagnostic
stratification.

Project implications:

The next PDE step should run a same-seed dense KAE control before any
multi-system sweep. The current model setting is good enough to use as the
LISTA candidate while checking whether support alignment and forecasting are
specific to LISTA rather than an artifact of the dataset.

## Same-Seed Dense Control Result

Concrete result:

- Jobs: dense train/eval `9553549`, dense support-threshold sweep `9553603`
- Dense control: `generic_no_shrink`, same one-seed dataset, same train/val/test split, same horizons, `TARGET_SIZE=512`, `SPARSITY_COEFF=0`
- Dense same-threshold read (`support_threshold=1e-4`, Jaccard `0.7`): `H=12` field MSE `0.885`, final-basin consistency `1.000`, `1` validation representative, `1` test family, purity `0.333`, NMI `0.000`, ARI `0.000`
- Dense best-NMI threshold read (`support_threshold=0.1`, Jaccard `1.0`): `H=12` field MSE `0.885`, final-basin consistency `1.000`, `224` validation representatives, `9` test families, purity `0.917`, NMI `0.616`, ARI `0.193`
- Matched LISTA reference (`support_threshold=1e-4`, Jaccard `0.7`): `H=12` field MSE `1.032`, final-basin consistency `0.917`, `7` validation representatives, `6` test families, purity `0.917`, NMI `0.709`, ARI `0.502`

Context:

The dense control was allowed the same support-family evaluation and an
additional threshold/Jaccard sweep. Labels remained evaluation-only for both
models.

Interpretation:

The result is mixed. Dense KAE forecasts the one-seed PDE field better than
LISTA at the tested horizons, so this pilot does not support a forecasting
advantage claim. LISTA has the cleaner basin-support alignment: it reaches
higher NMI and much higher ARI with only `7` validation representatives,
whereas dense needs exact-support-style overfragmentation (`224`
representatives) to approach the same purity and still has weaker ARI.

Project implications:

This benchmark is not ready as main paper evidence with the current flattened
models. It is justified as an appendix protocol and as a one-seed feasibility
pilot for basin-support alignment, but a paper-facing result should require
either a convolutional LISTA model, multiple seeds/systems, or a setting where
LISTA is not worse than dense on field rollout.

Next steps:

1. Keep all-held-out support alignment as the primary metric; use final
   majority fraction only as a diagnostic stratifier.
2. Do not scale this flattened run directly into a headline benchmark.
3. If this benchmark is pursued, implement the convolutional autoencoder path
   and rerun the same one-seed dense-vs-LISTA comparison before multi-system
   sweeps.

## Decisions

- First source system: `cal_square_4`.
- First smoke grid: `32x32` two-channel fields, observed dimension `2048`.
- First model path: existing `LISTAKM` through `lista_parity_generic_sparse`, using flattened fields. This is the closest existing LISTA path; convolutional encoders/decoders remain the next implementation step after the one-seed result runs.
- Paper-facing model path: convolutional Koopman autoencoders with
  LISTA-style shrinkage, dense encoder control, and sparse-MLP shrinkage
  control.
- Dataset storage: dependency-free `.pt` bundle for the smoke path. The generator also supports `.h5`/`.hdf5` when `h5py` is installed, but `h5py` is not currently a project dependency.
- Training uses only field sequences. Evaluation-only metadata includes attractor centers, per-pixel final basin maps, trajectory global basin labels, final majority fractions, and selected initial centers. Training does not consume basin labels, selected center indices, known basin count, or fixed support families.
- Main alignment objective remains basin-support alignment. No basin-block training design is introduced here.

## One-Seed Target

Default smoke command parameters:

- Source: `cal_square_4`
- Seed: `0`
- Grid: `32`
- Diffusion: `0.01`
- Integrator: RK4 `dt=0.01`, `10` substeps per stored observation
- Trajectories: `48` train, `12` validation, `12` test
- Stored length: `24` transitions
- LISTA: target size auto-resolves to the required `4*d_x` minimum
  (`8192` at grid `32`), `2` refinement loops, alpha `0.1` for the baseline
  smoke; tuned one-seed candidate alpha `0.001`
- Training: `500` steps, batch size `16`, sequence length `4`, prediction
  coefficient `1.0`

## How To Run

Submit the one-seed chain from the login node:

```bash
mkdir -p logs/slurm && sbatch scripts/run_spatialized_reaction_diffusion_one_seed.sh
```

Useful small overrides:

```bash
mkdir -p logs/slurm && GRID_SIZE=16 NUM_STEPS=100 TARGET_SIZE=2048 sbatch scripts/run_spatialized_reaction_diffusion_one_seed.sh
mkdir -p logs/slurm && GRID_SIZE=32 NUM_STEPS=1000 TARGET_SIZE=8192 sbatch scripts/run_spatialized_reaction_diffusion_one_seed.sh
```

Expected outputs:

- Dataset: `runs/spatialized_reaction_diffusion/cal_square_4_seed0_grid32_smoke/dataset.pt`
- Training run: `runs/spatialized_reaction_diffusion/cal_square_4_seed0_grid32_smoke/lista/`
- Evaluation: `runs/spatialized_reaction_diffusion/cal_square_4_seed0_grid32_smoke/evaluation.json`

## Evaluation Checks

The evaluation tool reports:

- Forecast `field_mse` and `final_field_mse` at horizons `1`, `4`, `8`, and `12`.
- Periodic-gradient MSE as a spatial smoothness/edge diagnostic.
- Same-horizon modal basin consistency by assigning predicted and true fields at
  the evaluated horizon to nearest attractor centers.
- Final-fate basin consistency by comparing the predicted modal basin at the
  evaluated horizon to the evaluation-only label-settled global fate.
- Pixel-level basin-map accuracy/IoU, majority-fraction error, and Fourier-band
  MSE.
- Support-family alignment where validation states define frozen Jaccard representatives, then test trajectories are assigned by nearest representative and compared to evaluation-only global basin labels.
- Alignment metrics: `H(B|F_abs)`, `H(F_abs|B)`, NMI, ARI, purity, validation representative count, and mean state-to-representative Jaccard.
- All held-out test trajectories are the primary support-alignment set.
- Final majority fraction is reported only as a diagnostic stratification variable, not as the main filter.

## Validation Performed

- Inspected `docs/appendix/benchmark_extensions.tex` and existing LISTA model/training APIs.
- Kept edits to PDE-specific benchmark utilities, PDE-specific CLI tools, one PDE SLURM runner, and this progress doc.
- Ran `bash -n scripts/run_spatialized_reaction_diffusion_one_seed.sh`.
- Re-ran `bash -n` and `git diff --check` after integration edits.
- Checked the new files for trailing whitespace with `rg -n "[[:blank:]]$" ...`; no matches.
- `shellcheck` is not available in the current shell.
- No Python, pytest, training, or evaluation was run on the login node.
- Submitted job `9530509`; it failed at evaluation serialization because the
  model metrics included a string diagnostic. Patched metric JSON conversion.
- Submitted job `9530539`; it completed with exit `0:0` and produced
  `evaluation.json`.
- Submitted tuning jobs `9530782`, `9530783`, and `9530784`; all completed
  with exit `0:0`.
- Submitted PDE threshold/Jaccard sweep jobs `9530780` and `9530799`; both
  completed with exit `0:0`.
- Submitted dense control train/eval job `9553549`; it completed with exit
  `0:0`.
- Submitted dense control support-threshold sweep job `9553603`; it completed
  with exit `0:0`.
- May 21 static shell check for new/updated SLURM scripts:
  `bash -n scripts/run_spatialized_reaction_diffusion_one_seed.sh scripts/run_spatialized_reaction_diffusion_array.sh scripts/queue_spatialized_reaction_diffusion_benchmark.sh`.
- Submitted tiny convolutional smoke jobs. Direct main-cpu job `9615803`
  completed with exit `0:0`. Task-table launcher `9615822` completed with exit
  `0:0` and submitted array `9615825`; all `3/3` child tasks completed with
  exit `0:0`. Earlier redundant pending smoke jobs `9615743`, `9615748`,
  `9615800`, and `9615802` were canceled after `9615803` completed.

## Blockers And Risks

- The convolutional path is runtime-validated only by a tiny smoke; this checks
  mechanics, not benchmark quality.
- HDF5 output requires `h5py`, which is not in `pyproject.toml` or `uv.lock`; the smoke default is `.pt` to avoid dependency churn before the first result.
- `32x32` is the confirmed flattened one-seed smoke. The convolutional path
  should validate at `8x8`/`16x16`, then `32x32`, before any `64x64` queue.
- Continuum Laplacian scaling makes the explicit RK4 stability margin depend on
  grid size. `64x64` should use smaller `rk4_dt` or weaker diffusion than the
  older flattened smoke defaults.

## Next Steps

1. Complete the tiny convolutional smoke and fix any runtime issues.
2. Rerun the one-seed dense-vs-LISTA-vs-sparse-MLP comparison with the
   convolutional model family.
3. Only after the one-seed convolutional result has both competitive forecasting
   and support alignment, decide whether to scale to `64x64`, multiple systems,
   or multiple seeds.
