# ManiSkill Insertion Benchmark Progress

## Current Status

Scope: contact-rich robotic insertion extension from
`docs/appendix/benchmark_extensions.tex`, first targeting state-only
`PegInsertionSide-v1`.

Implemented a minimal one-seed path:

1. Download/compact wrapper for state-only ManiSkill demonstrations, using raw
   `env_states` by default and simulator replay as an opt-in path.
2. Compact HDF5-to-NPZ dataset construction with trajectory-level splits.
3. Controlled LISTA/SKAE trainer for state/action windows.
4. Evaluator with rollout MSE, decoded-prediction periodic re-encoding, and
   frozen validation support-family alignment.

The latest corrected `5k` perturbation-balanced pilots run three seeds on the
`20`-source-episode packet. RGB-D remains deferred.

Prepare job `9530626` completed successfully and produced
`data/maniskill/PegInsertionSide-v1_state_compact_seed0.npz`. Controlled LISTA
train/eval job `9530627` also completed successfully. One-seed tuning and the
same-seed dense control are now complete. A first perturbation-balanced
simulator packet was generated and assessed with one seed per model setup on
May 19. This is a working state-only pilot, but the perturbation benchmark is
not ready to include as paper evidence yet because the current labels are still
perturbation-target labels, not validated contact/outcome labels. Later
corrected `5k` dense-tanh comparisons make the forecasting side more
promising, but they do not fix the semantic-label gap.

## 2026-06-09 Worker-A Protocol Audit

Claim under test:

Sparse Koopman autoencoders should form latent support families that align with
held-out contact/outcome regimes in contact-rich controlled insertion data,
while maintaining competitive forecasting. This is application-style evidence;
it should not be described as proof of true mathematical basins in ManiSkill.

What is currently implemented:

- Dataset generation:
  [tools/maniskill_generate_perturbed_rollouts.py](/home/mila/l/lia/skae/tools/maniskill_generate_perturbed_rollouts.py)
  replays `PegInsertionSide-v1` demonstration actions under five target
  perturbation families: `success`, `jam`, `miss`, `drop`, and `partial`.
- Compact dataset:
  [skae/benchmarks/maniskill_insertion_dataset.py](/home/mila/l/lia/skae/skae/benchmarks/maniskill_insertion_dataset.py)
  stores observations, actions, valid masks, train/val/test splits, outcome
  labels, optional contact-phase labels, episode ids, feature names, and
  metadata. The training-facing fields are observations, actions, valid masks,
  and splits.
- Label use:
  [tools/train_maniskill_controlled_lista.py](/home/mila/l/lia/skae/tools/train_maniskill_controlled_lista.py)
  samples only normalized state/action windows from the train split. Outcome
  and contact labels are evaluation-only metadata.
- Support evaluation:
  [tools/evaluate_maniskill_controlled_lista.py](/home/mila/l/lia/skae/tools/evaluate_maniskill_controlled_lista.py)
  builds greedy support-family prototypes from validation states, assigns
  held-out test states by nearest Jaccard similarity, and reports outcome and
  contact-phase alignment using conditional entropies, NMI, purity, ARI,
  support-family count, mean support size, mean family similarity, and
  unassigned-state fraction.
- Forecasting evaluation:
  the same evaluator reports no-reencode rollouts and optional
  decoded-prediction periodic re-encoding. Periodic re-encoding re-encodes the
  model's own decoded prediction, not the ground-truth future state.

Current packet:

- Dataset:
  [data/maniskill/perturbation_assessment_seed0_e20/all_setups.npz](/home/mila/l/lia/skae/data/maniskill/perturbation_assessment_seed0_e20/all_setups.npz)
- Source task: `PegInsertionSide-v1`
- Source demonstrations: `20`
- Total rollouts: `100`
- Target labels: `20` each for `success`, `jam`, `miss`, `drop`, `partial`
- Split rule: grouped by source episode, so perturbations of one reset do not
  cross train/validation/test.
- Actual final success audit from
  [perturbation_summary.json](/home/mila/l/lia/skae/data/maniskill/perturbation_assessment_seed0_e20/perturbation_summary.json):
  target `success` has `13/20` final successes, target `jam` has `2/20`, and
  `miss`, `drop`, and `partial` have `0/20`. These are therefore target
  perturbation labels, not validated physical outcome labels.

Current support-regime read from the June 9 fair optimizer run:

Under `support_threshold=0.2`, `family_jaccard=0.4`, `last.pt`, and three
seeds, the latest fair optimizer run gives the following mean held-out support
alignment diagnostics:

| Setting | Mean support families | Outcome NMI | Outcome purity | Contact NMI | Mean support size |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense tanh, no sparsity, `lr=5e-4,wd=0` | `7.33` | `0.115` | `0.267` | `0.0109` | `177.6` |
| Sparse MLP ReLU, sparsity `0.003`, `lr=5e-4,wd=0` | `31.0` | `0.466` | `0.556` | `0.0150` | `150.8` |
| Sparse MLP ReLU, sparsity `0.01`, `lr=5e-4,wd=0` | `53.0` | `0.628` | `0.711` | `0.0120` | `92.0` |
| LISTA ReLU, `alpha=0.01`, sparsity `1e-4`, standard optimizer | `3.67` | `0.120` | `0.244` | `0.0394` | `223.7` |

Interpretation:

Sparse MLP currently gives the clearest target-label support signal and is also
the strongest corrected small-pilot forecaster. LISTA is positive for
forecasting relative to dense tanh in the corrected fair comparison, but it is
not currently the best support-regime model on the perturbation packet. Contact
NMI should not be used as a claim yet because the perturbation-generation path
sets contact phase from success/non-success flags rather than from contact,
depth, distance, grasp, or rim geometry.

Missing before a paper-facing support-regime claim:

1. A semantic post hoc labeler that replaces target perturbation labels with
   actual physical outcomes: success, jam, miss, drop, partial, and ambiguous.
2. Timestep contact-phase labels derived from explicit geometric or simulator
   signals: peg-hole distance, insertion depth, grasp/drop state, sustained
   contact, rim contact, and final success.
3. A fixed support protocol selected without test labels: checkpoint rule,
   support threshold, Jaccard merge threshold, periodic re-encoding grid, and
   primary horizons.
4. A support summary artifact that aggregates outcome/contact metrics across
   seeds and reports family-count calibration beside forecasting.
5. A larger packet only after the semantic labeler passes an audit; otherwise
   scaling mainly increases confidence in target-perturbation artifacts.

Next SLURM plan for this workstream:

1. CPU or low-GPU label audit: generate a label-audit CSV for the existing
   `e20` packet using any available simulator/state features and keep
   ambiguous cases explicit. CPU is likely sufficient if it only reads stored
   states; request GPU only if ManiSkill simulator replay is needed for missing
   geometry/contact signals.
2. GPU packed training rerun after label audit: use the fair optimizer protocol
   with dense tanh, sparse MLP ReLU `sparsity in {0.003,0.01}`, and the best
   LISTA ReLU row. Keep three seeds first, then expand only if labels are
   semantically valid.
3. CPU evaluation/summarization: support sweeps and JSON aggregation should run
   CPU-only unless batched model inference becomes the bottleneck.

Recommended resources:

- Label audit without simulator replay: `long`, CPU-only, `4` CPUs, `16G`,
  `1` hour.
- Simulator-backed relabeling if needed: `long`, `1` GPU, `4`--`8` CPUs,
  `16G`--`24G`, `1`--`2` hours. This should be submitted only after checking
  that the relabeler actually steps ManiSkill on CUDA-capable nodes.
- Packed `5k`/`50k` model reruns: reuse the packed GPU launcher pattern with
  `1` GPU, `8` CPUs, `24G`, `<=3` hours per array task, `PACK_CONCURRENCY>=2`,
  and telemetry enabled. Prior telemetry for the packed `5k` fairness run was
  acceptable, but single-model GPU jobs would be wasteful because these models
  are small.

## 2026-06-10 Semantic Label Audit Result

Concrete result:

- Job: `9801691`, exit `0:0`
- Script:
  [run_maniskill_perturbation_label_audit.sh](/home/mila/l/lia/skae/scripts/run_maniskill_perturbation_label_audit.sh)
- Audit tool:
  [audit_maniskill_perturbation_labels.py](/home/mila/l/lia/skae/tools/audit_maniskill_perturbation_labels.py)
- Output:
  [label_audit_summary.json](/home/mila/l/lia/skae/results/maniskill_perturbation_label_audit_20260610/label_audit_summary.json)
  and
  [label_audit_rows.csv](/home/mila/l/lia/skae/results/maniskill_perturbation_label_audit_20260610/label_audit_rows.csv)
- Resources: CPU-only, `4` CPUs, `16G`; elapsed `17` seconds; no GPU was
  allocated.

Result:

- Target labels remain balanced: `20` each for `success`, `jam`, `miss`,
  `drop`, and `partial`.
- Actual final success by target is `success: 13/20`, `jam: 2/20`,
  `miss: 0/20`, `drop: 0/20`, and `partial: 0/20`.
- Available semantic relabeling from the current stored signals is only binary:
  `15` final-success rollouts and `85` non-success rollouts.
- The compact packet does not expose named geometry/contact signals needed for
  five-way physical relabeling: no contact, insertion-depth, peg-hole-distance,
  grasp/drop, or rim/alignment feature groups were detected.

Interpretation:

The current `e20` packet is usable for pipeline and target-perturbation
debugging, but not for a paper-facing claim that sparse supports discover
validated success/jam/miss/drop/partial regimes. The next paper-facing
insertion step is to regenerate or augment rollouts with explicit semantic
signals, then rerun the fixed support/forecast protocol.

## First Prepare Result

Concrete result:

- Job: `9530626`
- Compact dataset: `data/maniskill/PegInsertionSide-v1_state_compact_seed0.npz`
- Episodes: `1000`
- Split: `700` train, `150` validation, `150` test
- State dimension: `78`
- Action dimension: `8`
- Max transitions: `150`
- Evaluation labels: outcome labels available; coarse contact-phase labels available from feature-name heuristics

Context:

This result validates the state-only dataset interface without simulator replay.
The compact dataset is built from raw downloaded `env_states` and actions, with
previous action appended to each state. Training will still use only
observations, actions, transition masks, and splits.

Interpretation:

The prepare path is now usable for the one-seed LISTA training gate. It is not
yet the full benchmark protocol because it does not include perturbation-based
success/jam/drop/miss balancing and it avoids simulator replay on the current
Vulkan-limited node.

Project implications:

Keep robotics secondary until the one-seed LISTA support collapse is fixed.
The next robotics work is dense controlled KAE comparison and then
outcome-diverse perturbation generation on a node where ManiSkill
replay/simulation works.

## First One-Seed LISTA Result

Concrete result:

- Job: `9530627`
- Run directory: `runs/maniskill_insertion/controlled_lista_seed0`
- Checkpoint step: `1999`
- Best validation rollout MSE: `0.02559`
- Test state MSE: `0.000481` at `H=10`, `0.000838` at `H=25`, `0.006656` at `H=50`, `0.948929` at `H=100`
- Test final-state MSE: `0.000458` at `H=10`, `0.002022` at `H=25`, `0.035189` at `H=50`, `11.2738` at `H=100`
- Frozen validation support families: `1`
- Outcome alignment: `2` outcome labels available, but NMI/ARI `0.0` because all states map to one family; outcome purity `0.653`
- Contact-phase alignment: `2` phase labels available, but NMI/ARI `0.0` because all states map to one family; phase purity `0.951`

Context:

The run validates controlled LISTA training and evaluation on compact
state/action windows. Labels are not used for training. Support prototypes are
built on validation states and assigned to test states post hoc.

Interpretation:

The rollout model learns short-horizon state prediction on the compact
demonstration packet, but support discovery is degenerate at the current
threshold/hyperparameters: only one support family is found. Long-horizon
`H=100` final-state error also grows sharply. This is a working training path,
not evidence that supports recover contact/outcome regimes yet.

Project implications:

At this stage, the right response was threshold/sparsity tuning and a dense
controlled KAE comparator on the same compact dataset before any scale-up.

## One-Seed Tuning Result

Concrete result:

- Jobs: `9530785`, `9530788`, `9530789`, `9530790`; broad eval sweep `9530781` was cancelled after it became dominated by overfragmented high-Jaccard settings.
- Best rollout/support tradeoff so far: `LISTA_ALPHA=0.2`, `SPARSITY_WEIGHT=0.03`, support threshold `0.1`, Jaccard `0.7`
- Test state MSE: `0.000379` at `H=10`, `0.00650` at `H=50`, `0.2478` at `H=100`
- Outcome alignment: NMI `0.347`, purity `0.973`
- Contact-phase alignment: NMI `0.157`, purity `0.994`
- Support-family count at that read: `2017`, which is too fragmented for the intended paper diagnostic.
- Best compact read from follow-up sweep `9531152`: support threshold `0.2`, Jaccard `0.4`, `131` families, outcome NMI `0.303`, outcome purity `0.900`, contact-phase NMI `0.175`, contact-phase purity `0.985`
- Highest-NMI compressed read from `9531152`: support threshold `0.3`, Jaccard `0.5`, `546` families, outcome NMI `0.336`, outcome purity `0.940`
- More aggressive compression from the cancelled partial sweep: support threshold `0.05`, Jaccard `0.4`, `36` families, outcome NMI `0.288`, outcome purity `0.887`

Context:

The tuning changed LISTA shrinkage, sparsity weight, support threshold, and
Jaccard family merging only on the one-seed compact state dataset. Outcome and
contact labels remained evaluation-only.

Interpretation:

The original one-family collapse is fixed. The remaining robotics issue is
family-count calibration: higher Jaccard settings recover stronger outcome
purity but overfragment into hundreds or thousands of families. Lower Jaccard
settings give a more plausible family count with a moderate outcome/contact
signal.

Project implications:

The current one-seed robotics candidate is good enough to justify a dense
controlled KAE comparator on the same compact dataset, but not yet a broad
rollout-generation campaign. The next method-side tuning target is a
family-count/utility tradeoff rather than basic pipeline viability.

## Same-Seed Dense Control Result

Concrete result:

- Jobs: dense controlled KAE train/eval `9553550`, dense support-threshold sweep `9553602`
- Dense control: same compact dataset, same `700/150/150` split, same horizons, `ENCODER_KIND=dense`, `SPARSITY_WEIGHT=0`
- Dense same-threshold read (`support_threshold=0.2`, Jaccard `0.4`): H50 state MSE `0.00831`, H100 state MSE `3.644`, `1` family, outcome purity `0.653`, outcome NMI `0.000`, outcome ARI `0.000`, contact-phase NMI `0.000`
- Dense best-outcome-NMI read (`support_threshold=1.0`, Jaccard `0.7`): H50 state MSE `0.00831`, H100 state MSE `3.644`, `753` families, outcome purity `0.927`, outcome NMI `0.312`, outcome ARI `0.016`, contact-phase NMI `0.148`
- Matched LISTA compact read (`support_threshold=0.2`, Jaccard `0.4`): H50 state MSE `0.00650`, H100 state MSE `0.248`, `131` families, outcome purity `0.900`, outcome NMI `0.303`, outcome ARI `0.046`, contact-phase NMI `0.175`
- LISTA high-NMI read (`support_threshold=0.3`, Jaccard `0.5`): H100 state MSE `0.248`, `546` families, outcome purity `0.940`, outcome NMI `0.336`, contact-phase NMI `0.164`

Context:

The dense control used the same state/action windows and evaluation-only labels
as LISTA. The sweep checked whether dense support alignment was being hidden
by the default threshold.

Interpretation:

The robotics pilot is more promising than the PDE pilot. LISTA is better on
long-horizon rollout by a large margin at H100 and has useful outcome/contact
support alignment in the compact read. Dense can reach a similar outcome NMI
only by overfragmenting into `753` support families, and even then its H100
rollout is much worse and contact-phase NMI remains below LISTA.

Project implications:

This is not yet enough for a main paper benchmark because the dataset is still
raw demonstration `env_states`, has only two available outcome labels in the
current compact packet, and lacks perturbation-balanced success/jam/drop/miss
rollouts. It is a credible one-seed stage-one result that justifies building
the perturbation-balanced state-only dataset before adding RGB-D.

Next steps:

1. Generate perturbation-balanced state-only rollouts with explicit
   success/jam/drop/miss/partial outcome labels.
2. Keep all held-out states as the primary support-alignment set.
3. Rerun the same LISTA-vs-dense comparison before any RGB-D expansion.

## Perturbation-Balanced Rollout Plan

Purpose:

The next dataset should test policy-conditioned contact/outcome regime
discovery, not known mathematical basin recovery. ManiSkill does not provide a
ground-truth basin count. For a fixed scripted or demonstration-replay policy
and its perturbed action distribution, the evaluation question is whether
learned sparse support families predict held-out rollout dynamics and align
with post hoc outcome/contact labels.

Concrete rollout recipe:

1. Start from nominal `PegInsertionSide-v1` demonstrations or a scripted
   insertion controller.
2. For each reset/object geometry, replay the nominal action sequence and
   generate perturbed variants.
3. Record compact simulator states, actions, masks, reset metadata, and object
   geometry identifiers.
4. Append `20`--`50` settling steps with hold or zero action before assigning
   final outcome labels.
5. Split train/validation/test by reset seed, object geometry, and nominal
   source episode so held-out outcomes are not duplicated perturbations of the
   same exact setup.

Perturbation families:

| Target outcome | Perturbation type | Intended effect |
| --- | --- | --- |
| `success` | no perturbation or low action noise | clean insertion or near-demo behavior |
| `jam` | small lateral/yaw offset near approach or first contact | persistent rim contact without full seating |
| `miss` | larger lateral offset or approach-direction bias | no sustained peg-box contact |
| `drop` | gripper-open, weak-grasp, or grasp-timing perturbation | object leaves the gripper before insertion |
| `partial` | stop, hold, or retract near first contact | some insertion/contact without stable success |

Post hoc trajectory outcome labels:

| Label | Evaluation-only rule sketch |
| --- | --- |
| `success` | ManiSkill success flag is true, or final insertion depth and alignment exceed the task threshold after settling |
| `jam` | peg-box contact persists for at least `M` steps, success is false, peg tip is near the hole or rim, and insertion depth stays below success |
| `drop` | peg is no longer held, success is false, and the peg is low, table-contacting, or far from the insertion pose |
| `miss` | no sustained peg-box contact, success is false, and the peg tip remains far from the hole |
| `partial` | some contact or insertion depth is achieved, success is false, and final alignment/depth is unstable or below the seated threshold |

Balance target:

The first balanced state-only packet should be small enough for fast
one-seed tuning but large enough to avoid majority-class artifacts. A practical
first target is `100`--`250` rollouts per label (`500`--`1250` total), with no
label below roughly `15%` of the train/validation/test split. If a perturbation
family produces too many ambiguous cases, keep those rollouts in an
`ambiguous` audit bucket rather than forcing them into the five main labels.

Contact-phase labels:

When state features expose enough geometry/contact information, also assign
timestep labels post hoc:

- `free_space_approach`
- `grasped_transport`
- `alignment`
- `first_contact`
- `sliding_insertion`
- `jam`
- `seated_success`
- `drop_or_recovery`

These labels are used only after training to evaluate \(H(C_t\mid F_{\rm abs})\),
NMI, ARI, and phase purity. They are not used to construct support families,
choose support thresholds, select checkpoints, or train the model.

Diagnostic gate:

The perturbation-balanced packet is useful only if the three checks agree:

1. LISTA has competitive or better held-out rollout dynamics than dense
   controlled KAE, especially at H50/H100 on relative peg-hole state variables.
2. Validation-frozen support families align with held-out outcome/contact
   labels without requiring extreme family counts.
3. Dense controls cannot recover the same outcome/contact signal except by
   severe overfragmentation, and overfragmented dense supports still do not
   match LISTA rollout quality.

## First Perturbation-Balanced Assessment

Concrete result:

- Generation job: `9598665`, exit `0:0`
- Dataset: `data/maniskill/perturbation_assessment_seed0_e20/all_setups.npz`
- Source episodes: `20`
- Target rollout labels: `success`, `jam`, `miss`, `drop`, `partial`
- Total rollouts: `100`
- Split: grouped by source episode, `70` train, `15` validation, `15` test
- Observation dimension: `51`
- Action dimension: `8`
- Labels used for training: no
- Model jobs: LISTA `9598698`, dense controlled KAE `9598696`, sparse-MLP controlled KAE `9598697`; all exit `0:0`

The perturbation-target labels are not yet semantically clean. Final ManiSkill
success occurred in `13/20` nominal-success rollouts, `2/20` jam-targeted
rollouts, and `0/20` miss/drop/partial rollouts. Thus this packet is useful as
a simulator/training smoke test, but it does not yet establish a reliable
success/jam/drop/miss/partial outcome partition.

Aggregate held-out forecasting state MSE:

| Model | H10 | H25 | H50 | H100 |
| --- | ---: | ---: | ---: | ---: |
| LISTA controlled SKAE | `0.00369` | `0.00700` | `0.0496` | `15.91` |
| Dense controlled KAE | `0.00384` | `0.00379` | `0.0231` | `7.83` |
| Sparse-MLP controlled KAE | `0.00352` | `0.00437` | `0.0651` | `143.43` |

Final-state MSE shows the same long-horizon pattern: LISTA `238.36`, dense
`125.52`, sparse-MLP `2563.33` at `H100`.

Held-out target-label support diagnostics at support threshold `0.2`, Jaccard
`0.4`:

| Model | Families | Outcome NMI | Outcome purity | Contact-phase NMI |
| --- | ---: | ---: | ---: | ---: |
| LISTA controlled SKAE | `22` | `0.616` | `0.733` | `0.0099` |
| Dense controlled KAE | `1` | `0.000` | `0.200` | `0.0000` |
| Sparse-MLP controlled KAE | `15` | `0.498` | `0.400` | `0.0017` |

Context:

This run tests the assessment machinery requested for the benchmark: simulator
rollout generation, grouped held-out splits by source reset episode, one-seed
training for LISTA/dense/sparse-MLP setups, and validation-frozen support
families assigned to test states. The trainer still consumes only observations,
actions, masks, and splits.

Interpretation:

The benchmark is runnable but not paper-ready. LISTA discovers support families
that align with target perturbation labels better than the dense model, but it
does not beat dense controlled KAE on held-out long-horizon forecasting. The
current contact-phase label is also only a coarse success/non-success proxy, so
near-zero contact NMI should not be overinterpreted as a negative contact-regime
result. Conversely, the outcome NMI should not be overinterpreted as a positive
contact/outcome discovery result because the labels are still mostly the
intended perturbation class rather than validated physical outcomes.

Project implications:

This benchmark should remain a progress/protocol appendix item rather than a
main-paper result. It is ready for debugging the assessment pipeline, but not
ready for claims that sparse supports recover policy-conditioned contact
regimes. To become paper-usable, the dense baseline must no longer dominate
long-horizon forecasting, and the post hoc labels must be derived from actual
success, insertion depth, peg-hole distance, grasp/drop state, and contact/rim
geometry rather than target perturbation names.

Next steps:

1. Replace target labels with actual post hoc outcome rules and keep ambiguous
   rollouts in an audit bucket.
2. Add explicit state features for peg-hole distance, insertion depth, grasp
   state, and contact/rim contact when available from ManiSkill.
3. Retune only after the labeler is valid; initial knobs are LISTA sparsity
   weight and support threshold/Jaccard, but forecasting must remain the first
   gate.
4. Scale the state-only packet only after LISTA is at least competitive with
   dense forecasting on held-out source episodes.

## Decisions

- Use state-only observations first. RGB-D is intentionally deferred.
- Use controlled dynamics, `z_{t+1} = K z_t + B phi(a_t)`, rather than closed-loop autonomous modeling.
- Use a compact `.npz` dataset as the SKAE-facing interface so the trainer does not depend on the ManiSkill simulator. The compact state appends previous action by default; the controlled transition still receives current action separately.
- For the first smoke, compact raw downloaded `env_states` directly. Simulator
  replay with `USE_REPLAY=1` is deferred because the current CPU compute node
  fails ManiSkill render-system initialization before state replay starts.
- Keep outcome/contact labels evaluation-only. The trainer reads only observations, actions, transition masks, and train/val/test splits.
- Build support-family prototypes from validation states only, then assign test states to frozen representatives by nearest Jaccard similarity.
- If detailed contact labels are not available in replayed state features, report rollout metrics and outcome alignment from ManiSkill `success`/`fail` signals when present.
- The next state-only dataset should be perturbation-balanced across
  `success`, `jam`, `drop`, `miss`, and `partial`; ambiguous cases should be
  retained for audit but excluded from the primary five-label read.
- Split perturbation-balanced datasets by source episode/reset geometry so
  variants of the same underlying rollout do not appear in both train and test.

## One-Seed Target

Target run:

- Task: `PegInsertionSide-v1`
- Observation mode: `state`
- Control mode: source demonstration control mode by default. Set `CONTROL_MODE=pd_ee_delta_pose` only when explicitly testing converted end-effector actions.
- Dataset scale for smoke result: `COUNT=64`, `MAX_STEPS=150`
- Model: controlled LISTA, `z_dim=128`, `lista_loops=2`, `sequence_length=10`
- Training: one seed, default `SEED=0`, `NUM_STEPS=2000`
- Evaluation horizons: `10,25,50,100`

## How To Run

All Python execution must happen on a compute node. Submit scripts with `sbatch`.

Prepare a compact state dataset from raw downloaded `env_states`:

```bash
sbatch scripts/run_maniskill_insertion_prepare.sh
```

Opt-in simulator replay, only on a node/backend where ManiSkill replay works:

```bash
USE_REPLAY=1 OBS_KEY=obs sbatch scripts/run_maniskill_insertion_prepare.sh
```

If ManiSkill writes the replayed state trajectory under a non-standard name, rerun compaction with `REPLAY_TRAJ` set:

```bash
REPLAY_TRAJ=/path/to/trajectory.state.<control_mode>.<backend>.h5 \
OUTPUT=data/maniskill/PegInsertionSide-v1_state_compact_seed0.npz \
sbatch scripts/run_maniskill_insertion_prepare.sh
```

Run one-seed controlled LISTA training and evaluation:

```bash
DATASET=data/maniskill/PegInsertionSide-v1_state_compact_seed0.npz \
sbatch scripts/run_maniskill_insertion_one_seed.sh
```

The expected evaluation summary is:

```text
runs/maniskill_insertion/controlled_lista_seed0/eval_test/metrics_summary.json
```

## Validation Checks

Static/local checks completed:

- Added only ManiSkill-specific modules/scripts and this progress doc.
- Inspected the official ManiSkill dataset/replay documentation. Relevant facts: raw demos usually omit observations; replay with `-o state --save-traj` creates observations; HDF5 trajectories store `actions`, optional `obs`, optional `success`/`fail`, and metadata JSON needed to recreate tasks.
- `bash -n scripts/run_maniskill_insertion_prepare.sh scripts/run_maniskill_insertion_one_seed.sh`
- Whitespace check for all new ManiSkill files via `git diff --check --no-index /dev/null <file>`.
- Integration check found that the base project environment does not include
  `mani_skill`; prepare job `9530510` failed at import before data generation.
  The prepare runner now invokes ManiSkill/HDF5 steps through
  `uv run --with mani_skill --with h5py python`.
- Resubmitted prepare job `9530523`; one-seed controlled LISTA train/eval job
  `9530524` is queued with `afterok:9530523`.
- Job `9530523` reached the current ManiSkill CLI, downloaded the demo, and
  failed because `replay_trajectory` no longer accepts `--num-procs`. The
  runner now removes that option and points `RAW_TRAJ` at the downloaded
  motion-planning trajectory under `~/.maniskill/demos`.
- Job `9530538` then reached simulator replay and failed before replay because
  SAPIEN/Vulkan render-system initialization was unavailable on the assigned
  CPU node. The prepare runner now compacts raw `env_states` by default and
  keeps replay behind `USE_REPLAY=1`.
- Job `9530626` completed the raw-`env_states` compact dataset path with exit
  `0:0`; dependent train/eval job `9530627` is pending.
- Job `9530627` completed controlled LISTA train/eval with exit `0:0` and
  wrote `runs/maniskill_insertion/controlled_lista_seed0/eval_test/metrics_summary.json`.
- Submitted tuning jobs `9530785`, `9530788`, `9530789`, and `9530790`; all
  completed with exit `0:0`.
- Cancelled broad eval sweep `9530781` and targeted eval sweep `9530870` after
  they spent most time on high-Jaccard overfragmented settings. Compressed
  follow-up sweep `9531152` completed with exit `0:0`.
- Submitted dense controlled KAE train/eval job `9553550`; it completed with
  exit `0:0`.
- Submitted dense control support-threshold sweep job `9553602`; it completed
  with exit `0:0`.
- GPU ManiSkill smoke job `9598599` completed with exit `0:0`; the simulator
  can instantiate `PegInsertionSide-v1` in state mode and step on a GPU node.
- Perturbation generation job `9598665` completed with exit `0:0` and wrote
  the grouped `20`-source-episode packet under
  `data/maniskill/perturbation_assessment_seed0_e20/`.
- One-seed CPU model assessment jobs `9598698`, `9598696`, and `9598697`
  completed with exit `0:0` for LISTA, dense, and sparse-MLP setups.

Checks still needed on a compute node:

- Python import/syntax smoke for the new modules through `uv run`.
- A small compaction run from an actual replayed `PegInsertionSide-v1` state HDF5 file.
- The June 10 available-signal audit is complete and shows that the current
  packet is not sufficient for five-way physical outcome/contact labels. A
  further semantic relabeling audit is still needed after regenerating or
  augmenting rollouts with actual contact/depth/grasp/drop features rather than
  perturbation-target names.

## Blockers / Risks

- ManiSkill may save replayed state HDF5 files with a name that the prepare wrapper cannot auto-discover. `REPLAY_TRAJ` is the explicit override.
- The current `pyproject.toml` does not add ManiSkill as a project dependency. The prepare job uses per-command `uv run --with mani_skill --with h5py`; this avoids dependency churn but may spend time resolving/downloading ManiSkill on the compute node.
- Detailed contact-phase alignment depends on state feature names exposing contact, insertion depth, or peg-hole distance signals. If raw `env_states` do not expose those semantic names, the first smoke may report only rollout metrics and any available success/fail outcome labels.
- End-effector action conversion may fail for some demonstrations. The wrapper therefore uses the source control mode unless `CONTROL_MODE` is explicitly set.
- Simulator replay/generation works on the tested GPU path. CPU-only ManiSkill
  replay remains risky because earlier CPU-node replay failed during
  SAPIEN/Vulkan initialization; raw-`env_states` compaction is still the
  stable non-simulator fallback.
- The first perturbation packet is not cleanly balanced by actual outcome:
  nominal success succeeds in `13/20` episodes, while the non-success target
  classes are mostly failures but not yet separated into validated jam, miss,
  drop, and partial outcomes.

## Next Steps

1. Replace perturbation-target labels with validated post hoc outcome labels.
2. Add or extract contact/depth/distance/grasp features needed for timestep
   contact-regime labels.
3. Rerun one seed of LISTA, dense, and sparse-MLP after the labeler is valid;
   keep support threshold `0.2`, Jaccard `0.4` as the first compressed LISTA
   read unless forecasting or family counts clearly require retuning.
4. Only after the perturbation-balanced state-only comparison is positive,
   decide whether to add RGB-D.
