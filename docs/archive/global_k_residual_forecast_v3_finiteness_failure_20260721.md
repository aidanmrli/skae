# Global-K residual forecast V3: incomplete after nonfinite payload value

Date: 2026-07-21

## Concrete result

The prospectively frozen V3 experiment produced **no complete scientific
decision**. Array tasks 0--4 completed and remained quarantined. Task 5
completed its GPU compute window but failed while writing its permission-0600
scientific shard because strict JSON serialization encountered a non-finite
payload scalar (`inf`). Consequently, task 5 has no shard, the exact ten-seed
roster is incomplete, and the dependent scientific telemetry gate and summary
never ran. Tasks 6--9 and the two now-unreachable dependent jobs were cancelled
without allocation.

This is **invalid at the frozen validity tier and directionally unadjudicated**,
not negative evidence for the proposed
support-routed residual one-operator hypothesis. No partial shard, finite
prefix, surviving arm, or subset of model seeds may be scored or cited as a
performance result.

The GPU execution itself was efficient. The outcome-blind smoke gate passed on
an A100-80GB with 99.98% mean compute-window utilization, 74.69% allocation-wide
utilization, and 53,883 MiB peak memory. For science tasks 0--5, compute-window
means were 97.21--98.62%, allocation-wide means were 89.35--92.09%, and frozen
30-second rolling minima were 88.93--95.93%; all six individually cleared the
telemetry thresholds before task 5's serialization failure.

## Experimental context

The diagnostic used the ten paired sparse/exact-dense tanh checkpoints from the
controlled three-basin experiment. Support families were fitted and assigned
without basin labels or a known basin count. The main autonomous predictor
reencoded and rerouted at every physical step while using one unchanged learned
matrix:

\[
x^+ = x + D\!\left(((E(x)P_f)K)P_f\right)-D\!\left(E(x)P_f\right).
\]

It was frozen against unchanged-global, residual-global, routed-nonresidual,
32 matched coordinate-permutation, persistence, and exact-dense comparators at
H200 (physical time 8), with H500 (physical time 20) as a stress tier. Each of
three new evaluation corpora contained 131,072 trajectories. This predictor is
an autonomous nonlinear composite using one unchanged \(K\), not a pure
\(K^h\) Koopman rollout and not an invariant-subspace test.

The card required complete finite H200 endpoints, all ten model seeds, all
three datasets, and a dependent first-readable summary. It explicitly forbade
finite-prefix scoring, non-finite replacement, trajectory omission, threshold
changes, or treating missing/divergent arms as wins.

## Interpretation

The failure is not evidence that the sparse or dense arm wins, nor evidence
against the forecasting hypothesis. The traceback identifies neither the arm
nor endpoint and cannot distinguish a non-finite predicted state from overflow
of a metric computed from finite float32 states. The first-readable summary was
never produced. What it does establish is that this frozen execution did not
yield the complete, finite, authenticated packet required for any claim.
Whether the non-finite value arose in a confirmatory or context-only arm cannot
rescue the packet: the missing task-5 shard alone violates roster and
completeness requirements.

The unadjudicated run therefore leaves the earlier conclusion unchanged. Existing
checkpoints show unusually low support-projected one-step leakage relative to
matched coordinate nulls and exact-dense top-\(k\) coordinates, but the
operator-differentiation, distinct-law, and now multistep routed-residual tests
do not justify a learned one-matrix/multiple-local-laws claim.

## Project implications

- Do not promote this diagnostic as positive or negative forecasting evidence,
  and do not inspect or report the five completed partial shards.
- Retain the exact direct-sum affine construction only as a representational
  possibility result, clearly separated from learned-model evidence.
- Lead the forecasting rebuttal with the complete Lorenz--96 and Allen--Cahn
  sparse-versus-exact-dense results. Treat global-\(K\) support closure as a
  bounded one-step diagnostic and disclose its failed differentiation guard.
- Do not tune, repair, or rerun this mechanism test after observing the
  non-finite failure; doing so would add outcome-selection risk without filling
  the paper's highest-priority empirical gap.

## Next steps

No fourth attempt is authorized. Preserve the V3 root read-only, record the
failure in the rebuttal dossier, and keep the paper's claim boundary at lower
support-projected one-step leakage rather than invariant subspaces, distinct
local laws, or a decoded multistep mechanism.

## Provenance

The immutable output root is
`/network/scratch/l/lia/skae/global_k_residual_forecast_v3_20260721/`.
The card's operative `freeze.launch_authorized` field is true and the queue
required an authorization token bound to all three freeze roots. Its stale
top-level status still says that launch was pending; this is a transcription
defect, not evidence that the run lacked authorization. The V2 archive's
one-final-attempt rule governs, so no V4 ambiguity branch is available.
Card/task/source-manifest/queue SHA-256 values are
`fdb48269a6a0f7f964fcbf27271f54a67f195f6ef46d2e5c83ebcf67046629ca`,
`86a3dce2ce8fd6ca569aebcccb6812ac6c3ee206ec21ba8e2ccf2642305fb024`,
`2c7439ca57c61e74c9f05b1dbb4d9f9c19c0e32efe60587063e27ae4ab8bd8e8`,
and `db0222b88401214a34010e67ef0fdbf07d5d36d3ba9bc763249451a42afff8d4`.
The outcome-free data manifest and passed smoke assessment hashes are
`45db7977b350f592087c41f603e4056e930b2095bd0bb97664c857da4f801a99`
and `f2d256d76ab2fbba88c7a508ba846394bde6a3f268fd547e82801f05dea2f704`.
Task-5 (model seed 105) stderr, compute-window marker, telemetry trace, and stdout hashes are
`6d0e75d4dfc38eb494ed6056691d6e538b9e0e504ce483ebd3d61933f8bd0f6a`,
`5f7f09a3be60edc1cff98b5e137c3e067b93486b503a3f8355e1e0d1f7c54c2d`,
`8fbe9a69392a4aca9b3fb0a27300ba939536cbcfe98c0d2f54e37e123c19f1d4`,
and `fd7e862626f361b7afbaf1ab2f34a1d6420d134b228a1fba2af8ac624946b8f3`.

Jobs were queue `10165800`, prepare `10165803`, smoke `10165804`, smoke gate
`10165805`, science array `10165806`, science gate `10165807`, and summary
`10165808`. Array children 0--4 completed; child 5 (`10165847`) failed with
exit code `1:0`; children 6--9 and jobs `10165807`/`10165808` were cancelled.
