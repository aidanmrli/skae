# Transition-rich basin partitioning plan

---

NOTE: Update this file, `docs/EXPERIMENTS.md`, and `docs/PAPER_TRACK_STATUS.md` together.

Companion design-inventory file for the current restored worktree:
[docs/planning/transition_rich_system_inventory_20260406.md](/home/mila/l/lia/skae/docs/planning/transition_rich_system_inventory_20260406.md).

Companion audit of the already implemented Claude catalog:
[docs/planning/claude_catalog_audit_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_audit_20260407.md).

Active scope note:

The branch experiment scope is now frozen. Forward interpretability
experiments should use only the fixed `17`-system shortlist already recorded in
[docs/EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md) and
[docs/PAPER_TRACK_STATUS.md](/home/mila/l/lia/skae/docs/PAPER_TRACK_STATUS.md):

- native transition-rich trio:
  `multiwell_strong_transition`, `gated_local_linear`,
  `gated_transfer_linear`
- Claude-catalog subset:
  `arrested_spiral`, `cal_asymmetric_3`, `cal_high_cross_3`,
  `cal_hexagon_6`, `cal_octagon_8`, `cal_pentagon_5`, `cal_square_4`,
  `checkerboard_potential`, `duffing_triple_well`, `snic_multi`,
  `transition_routes_4`, `var_depth_gradient_4`, `var_diamond_4`,
  `var_l_shape_5`

Do not use this plan document to reopen broader system-generation or
system-selection scope for the current branch.

## Motivation

The difficulty is that the most interesting nonlinear systems typically contain multiple equilibria or multiple basins of attraction, and theory shows that for these systems with multiple basins of attraction, finite-dimensional Koopman-invariant subspaces cannot provide a globally exact linear representation across all basins simultaneously. In a multistable finite-dimensional system, we might be able to linearize the dynamics within each basin of attraction, but the linearizations needed in different basins of attraction are incompatible with each other. If Koopman models are to serve as a foundation for interpretable multi-regime modeling, then the latent representation should do more than forecast well: ideally, it should organize the state space into meaningful local dynamical regimes.

## Objective

We have sufficient evidence for forecasting ability in the paper. We now want
to focus on a new branch of experiments centered on **local basin partitioning
and classification** on deterministic toy systems.

The system set for this branch is no longer open-ended. The objective is to use
the fixed shortlist above as the full forward experiment scope, not to keep
expanding the candidate pool.

The branch should do the following:

1. Treat the fixed `17`-system shortlist as the full branch scope and avoid
   adding new systems unless the branch objectives are explicitly changed in the
   live docs. (DONE)
2. Plot these systems so that we can visually inspect them. (DONE)
3. Investigate whether different kinds of Koopman autoencoder can discover reusable basin partitions without training-time basin labels. This involves training and evaluating the various kinds of Koopman autoencoders on these systems. For now, we should use LISTA-based encoders for the Koopman autoencoder, and try both dense Koopman transition matrices and block-diagonal Koopman transition matrices. In this case, it is appropriate to use the ground truth number of basins to set the number of blocks, but keep the latent size at 256. We will expand this analysis to MLP encoders once we have good performance on LISTA-based encoders. 
**IMPORTANT NOTE:** When we evaluate the model on unseen trajectories, I want to save the ground truth trajectories AND the predicted trajectories so that we can do diagnosis on the performance of the model in part 4.
4. Create, inspect, and diagnose metrics that explain **why** forecasting succeeds or fails on these systems instead of just reporting MSE. These metrics may diagnose phase-plot crossings, partition reuse, support structure, and transition handling. We should reuse the cached 
5. Loop between (3) and (4) on the fixed shortlist.

The lead open question is whether the learned Koopman representation from the
encoder identifies meaningful partitions and handles common transitions cleanly
across this fixed shortlist.

## Tests first

No system-specific environment code or metric code should be written until the test files below exist in `tests/`.

| Planned test module | What it must verify |
| --- | --- |
| `tests/test_transition_rich_system_determinism.py` | Fixed config plus fixed initial state and any deterministic noise always produce identical trajectories |
| `tests/test_transition_rich_crossing_metrics.py` | Region-path parsing, crossing counts, and crossing-fraction summaries are correct on hand-constructed paths |
| `tests/test_transition_rich_env_registry.py` | New environment names construct through the existing factory |
| `tests/test_transition_rich_calibration_schema.py` | Calibration summaries always emit the required per-trajectory and per-system fields |
| `tests/test_transition_rich_num_transitions.py` | The number of trajectories that have at least one transition between basins/regions should be between 30-70% |

The required implementation order is fixed:

1. Write correct tests and verify that they fail.
2. Add only the minimum interface scaffolding needed for those tests.
3. Implement system-specific dynamics.
4. Implement calibration summaries.
5. Run model experiments only after the tests pass.

## Calibration and selection protocol

### Definitions

- **Endpoint basin**: the long-rollout attractor identity used only for benchmark evaluation.
- **Transition**: a finite-horizon change in nearest-center or dominant-region assignment along a trajectory.
- **Crossing trajectory**: a trajectory that visits at least one non-endpoint region before entering its settling window.

### Candidate-system screening

Before any model training on a shortlisted system, screen candidate parameters
offline from a fixed initial-condition box.

Each candidate must be evaluated on:

- endpoint-basin count,
- endpoint-basin occupancy,
- region paths,
- crossing fractions.

For the current branch, "candidate" means parameterizations or calibration
variants of the already chosen systems above, not wholly new benchmark-system
ideas.

### Acceptance gates

| Gate | Requirement |
| --- | --- |
| Determinism | System is deterministic and reproducible from fixed config plus fixed initial state |
| Plotability | Native `2D` phase portrait is clean enough for paper figures |
| Basin count | Between `3` and `10` endpoint basins |
| Basin occupancy | No endpoint basin below `max(0.05, 0.5 / B)` occupancy, where `B` is the endpoint-basin count |
| Transition richness | Per-endpoint-basin crossing fraction in the acceptable `0.30-0.70` range for the frozen first-pass pair |

If strict per-endpoint-basin `0.30-0.70` cannot be met, reject the candidate unless:

- overall crossing remains high,
- no endpoint basin falls below `0.25` crossing fraction,
- and the system still has clean deterministic mechanics worth studying in depth.

## Metric roadmap

MSE is not enough for this branch. The diagnostic stack must tell us whether the model learned reusable partitions, whether those partitions are predictive, and whether failures happen inside regimes or at transitions.

### Primary label-light metrics

These are the first metrics to report in the new branch because they do **not** require the true basin count as a primary input.

| Metric family | Metrics | Why it matters |
| --- | --- | --- |
| Support-group reuse | recurring support-group count, retained trajectory coverage | Measures whether the representation yields reusable partitions rather than one-off supports |
| Predictive usefulness | local-vs-global fit improvement, local-vs-shuffled fit improvement | Tests whether discovered partitions carry local dynamical information |
| Transition specificity | cross-over-within error ratio | Tests whether a support-defined model works better inside its own group than outside it |

These metrics should reuse the recurring-support local-linearity tooling wherever possible.

### Transition diagnostics

These metrics explain whether failures come from transition handling.

| Metric | Purpose |
| --- | --- |
| Crossing fraction | Measures how transition-rich the toy system actually is |
| First-exit rate | Measures how often trajectories leave their starting region |
| Dwell-time distribution | Measures how long trajectories stay in a region before switching |
| Rollout endpoint preservation | Measures whether predicted trajectories end in the correct endpoint basin |
| Flow-branching rate and severity | Measures whether forecast rollouts assign materially different futures to nearly identical full states |

For the native-plot `2D` deterministic candidates, add one explicit forecast-side flow-consistency diagnostic:

- use a deterministic initial-condition grid,
- pool forecasted full states across trajectories and rollout times,
- sweep a normalized same-state tolerance,
- derive the next-step divergence threshold from the ground-truth simulator so the true system is exactly zero by construction under the chosen sweep,

This diagnostic is not the headline metric, but it is important because the motivating systems are deterministic. Learned local partitions should not come at the cost of obvious forecast-side branching from effectively identical states.
