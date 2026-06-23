# Reframed Local-Law Mechanism Plan

Date: April 20, 2026

## Decision

We are reframing the fixed-`17` local-law mechanism study around the actual
paper question:

1. The main causal control is the dense no-sparsity MLP trained with the
   `generic_no_shrink` recipe, not the sparse MLP controls.
2. The mechanism claim is not "some post-hoc local zero-intercept fit beats one
   global zero-intercept fit."
3. The mechanism claim we actually need to test is:
   sparse support or support families identify a local chart, and conditioning
   on that chart selects a more appropriate local Koopman law than the dense
   global alternative.

The evaluation therefore has to separate three different effects:

- basin or chart offset
- local slope or Jacobian
- direct support-gating of the learned global Koopman matrix

## Why The Old Read Was Insufficient

The earlier operator-selection packet fit raw-coordinate zero-intercept local
operators,

`z_{t+1} \approx A_s z_t`,

and compared them against one global zero-intercept fit. That is not the right
test of a local linearization mechanism because a local chart centered at
`c_s` is generally affine in the ambient latent coordinates,

`z_{t+1} - c_s \approx A_s (z_t - c_s)`,

which is equivalent to

`z_{t+1} \approx A_s z_t + (I - A_s) c_s`.

So the old packet mixed together:

- slope differences
- chart-origin / offset differences
- support-gating effects

That means a positive result could come from local offsets rather than support
selecting a different local slope, while a negative result could come from
evaluating the wrong chart representation.

## New Evaluation Principle

For mechanism claims, use the following separation:

1. **Causal model comparison**
   Compare LISTA roots against the dense no-sparsity `tanh` control
   `mlp_zero_sparse_basin_partition_control`.

2. **Centered local-law evaluation**
   Evaluate local slopes in centered coordinates:
   `z_{t+1} - c_s \approx A_s (z_t - c_s)`.

3. **Support-gated Koopman evaluation**
   Evaluate whether the learned global Koopman matrix already behaves like a
   support- or family-selected local law when only the active support, or the
   active block set, is allowed to drive the prediction.

4. **Depth-stratified evaluation**
   Report all mechanism metrics across state-space depth strata, not only one
   pooled number. Local linearization should be strongest deep in a basin and
   weakest near separatrices.

## Experiment Matrix

### Roots

- `lista_dense_softblock_signsplit_p64_hardinit_basin_partition`
- `lista_blockdiag_signsplit_hardinit_basin_partition`
- `mlp_zero_sparse_basin_partition_control`

The sparse MLP controls remain useful for architecture-only side reads, but the
main mechanism queue should compare LISTA against the dense no-sparsity MLP.

### Support Definitions

- `relative:0.1`
- `topk:8`

These are the right default definitions for the mechanism packet because they
remain meaningful even on the dense `tanh` control where exact zeros are not
expected.

### Partitions

- `basin`
- `family`
- `support`

### Depth Strata

Use margin-to-separatrix quartiles on the current state:

- `q1`: boundary-adjacent
- `q2`
- `q3`
- `q4`: deepest states
- `all`

This preserves the existing deep-versus-boundary story while making the
mechanism read explicitly state-space dependent.

### Transition Regimes

- `all_current`: all transitions whose current state lies in the chosen depth
  stratum
- `persistent_current`: only transitions that stay in the same basin/support/
  family one step later

This cleanly separates chart validity from actual chart switching.

## Experiment 1: Centered Local Slope Study

For each root, support definition, depth stratum, transition regime, and
partition kind:

1. Fit one global centered slope on the training transitions:
   `z_{t+1} - c \approx A (z_t - c)`.
2. Fit one centered slope per basin/support/family on the same latent
   coordinates:
   `z_{t+1} - c_s \approx A_s (z_t - c_s)`.
3. Compare held-out latent one-step MSE against:
   - the learned global Koopman matrix
   - the global centered slope
   - random count-matched partitions
   - latent k-means partitions of equal class count

Primary question:

- does centered local conditioning beat the dense global alternative where it
  should, especially in `q4` and especially under `persistent_current`?

## Experiment 2: Support-Gated Koopman Mechanism Study

Using the learned global Koopman matrix from each checkpoint, evaluate
class-conditioned predictions with:

1. **input-gated support**
   Only the active support coordinates of `z_t - c_s` are allowed to drive the
   next-step prediction.
2. **support principal submatrix**
   Restrict both the source and target coordinates to the current support.
3. **block-gated variant**
   When the model exposes a block layout, activate the union of blocks touched
   by the current support prototype.

These tests directly answer the mechanism question:

- does support merely correlate with the basin, or does it actually select the
  operative part of the learned Koopman dynamics?

Primary comparison:

- LISTA roots versus the dense no-sparsity `tanh` control on the same gating
  diagnostics.

## Experiment 3: Prototype And Basin Canonicalization

For support-gated `K` reads:

- exact-support partitions use the exact support mask
- family partitions use the family prototype mask
- basin partitions use the canonical support mask inferred from deep (`q4`)
  states within each basin

This avoids defining a basin-level gating mask from noisy boundary states.

## Experiment 4: Self-Routed Forecasting Without Oracle Basin Labels

This is the deployment-facing packet that follows the centered-chart mechanism
study.

The rollout router must use only the model's own current latent support; it
must not use basin labels or a fixed basin count at forecast time.

### Rollout modes

- `global_k`
- `support_gated_k`
- `support_block_gated_k`
- `support_local_centered`
- `family_local_centered`

### Evaluation principle

1. Fit local support- and family-conditioned centered operators on separate
   fit trajectories from the same checkpoint.
2. Start forecasting trajectories from initial conditions spread across the
   whole state space.
3. Stratify forecast starts by basin-depth quartile (`q1` through `q4`) so we
   can read the expected deep-versus-boundary behavior directly.
4. At each rollout step, infer the current support from the model's own latent
   state and use that to choose the local law or support-gated Koopman mask.
5. Fall back to the global `K` if a support or family route is unavailable.

### Primary comparison

- LISTA roots versus the dense no-sparsity `tanh` MLP control
- self-routed local rollout versus one global `K`
- exact support versus support family as the routing signal

### Main outputs

- `H100`, `H500`, `H1000` rollout error
- depth-stratified rollout error (`q1` vs `q4`)
- route coverage / fallback fraction
- route switch rate / chatter
- skipped-mode accounting when support classes are too fragmented

### Why this is the real forecasting test

The centered-chart packet shows whether local laws exist once the chart is
centered. This self-routed forecasting packet is the actual paper-facing test
of whether the encoder can *discover and use* those local laws without oracle
basin routing.

## Expected Interpretation

### Positive mechanism evidence

The local-law story is strengthened if:

- centered local slopes beat the learned global `K` and the global centered
  slope in deep states, especially under `persistent_current`
- LISTA support/family partitions beat random and latent-kmeans controls in the
  same centered evaluation
- support-gated or block-gated `K` remains competitive with the centered local
  slope for LISTA but not for the dense no-sparsity MLP

### Negative mechanism evidence

The strong story should be dropped if:

- centered local slopes do not improve on the global learned `K`
- gains only appear from chart offsets and disappear after centering
- support-gated `K` is not materially better for LISTA than for the dense
  no-sparsity control

## Queue Plan

1. Implement one new centered-mechanism evaluator that reuses the existing
   run-discovery, trajectory generation, basin labeling, support-family
   clustering, and block-layout helpers.
2. Run a smoke packet on:
   - `gated_local_linear`
   - `gated_transfer_linear`
   - `claude:cal_square_4`
   with seed `0`.
3. If smoke passes, submit the full fixed-`17` packet over the three roots
   above, sharded by root, with a merge job.
4. After that packet is complete, run the self-routed forecasting packet on
   the same root trio and fixed-`17` checkpoint sources, again sharded by
   root, so the non-oracle forecasting read uses exactly the same model set.

## Current Implementation Status

- The centered-chart mechanism packet is complete.
- The self-routed forecasting packet is now implemented in
  [tools/evaluate_transition_rich_self_routed_forecasting.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_self_routed_forecasting.py)
  with merge utility
  [tools/merge_transition_rich_self_routed_forecasting_shards.py](/home/mila/l/lia/skae/tools/merge_transition_rich_self_routed_forecasting_shards.py)
  and SLURM launchers
  [scripts/run_transition_rich_self_routed_forecasting.sh](/home/mila/l/lia/skae/scripts/run_transition_rich_self_routed_forecasting.sh),
  [scripts/merge_transition_rich_self_routed_forecasting_shards.sh](/home/mila/l/lia/skae/scripts/merge_transition_rich_self_routed_forecasting_shards.sh),
  and
  [scripts/queue_transition_rich_self_routed_forecasting_shards.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_self_routed_forecasting_shards.sh).
- Smoke validation is complete under
  [results/transition_rich_self_routed_forecasting_smoke_20260420](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_smoke_20260420)
  with `270` rows and `0` failures across `9` reduced runs, and merge-path
  validation is complete under
  [results/transition_rich_self_routed_forecasting_merge_smoke_20260420/merged](/home/mila/l/lia/skae/results/transition_rich_self_routed_forecasting_merge_smoke_20260420/merged).
- The packet is launch-ready; the full SLURM submission is the next step once
  we explicitly sign off on the run.

## Paper-Facing Consequence

Until this reframed packet is complete, we should stop treating the earlier
zero-intercept local-fit results as direct evidence that support selects local
linear laws. The new packet is the decision-grade mechanism study for that
claim.
