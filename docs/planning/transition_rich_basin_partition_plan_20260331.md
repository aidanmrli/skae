# Transition-rich basin partitioning plan

---

## Objective

We have sufficient evidence for forecasting ability in the paper. We now want to focus on a new branch of experiments centered on **local basin partitioning and classification** on deterministic toy systems.

The branch should do the following:

1. Generate toy nonlinear systems in 2D and/or 3D that have interesting non-trivial transitions between basins; each system should procedurally generate movement between basins of attraction rather than just collapsing to the local attractor from the initial condition. These toy systems should be well-characterized and we should be able to plot them. These systems might be generated with deterministic noise, but there can be freedom in generating these systems.
2. Plot these toy systems so that we can visually inspect them.
3. Create, inspect, and diagnose interesting and possibly new metrics that explain **why** forecasting succeeds or fails on these toy systems instead of just reporting MSE. These metrics might diagnose whether phase plots are crossing (violations since each starting point should have its own unique solution). They might also study the extent to which distinct basins in our toy systems correspond to unique supports, perhaps by studying activations. Creativity in coming up with different metrics is encouraged.
4. Investigate whether different kinds of Koopman autoencoder can discover reusable basin partitions without training-time basin labels. A strong result would be if a LISTA-encoder model can perform better
5. Loop between (3) and (4).

The lead open question is whether the learned Koopman representation from the encoder identifies meaningful partitions and handles common transitions cleanly.

## Locked decisions

| Topic | Locked choice |
| --- | --- |
| Toy-system strategy | Reuse and extend `multiwell`, plus add deterministic chart-switching families |
| Basin-count range | Accept systems with `3-10` endpoint basins |
| Transition richness | For the frozen first-pass pair, use `0.30-0.70` as the acceptable per-endpoint-basin crossing-fraction range; for the explicit transfer family, use source-neighborhood transfer plus core-retention gates |
| Preferred transition target | Aim for approximately `0.50` when tuning whichever transition metric is active for the family |
| System type | Deterministic only; no stochastic forcing, random switching, or black-box label generation |
| Scientific depth | Prefer systems with explicit attractor centers, visible transition corridors, and analyzable local mechanics |
| Plotability | First milestone requires native `2D` systems; native `3D` is second pass |
| Training-time supervision | No basin labels or known basin counts in training-time method design |
| Evaluation-time supervision | Known endpoint labels and basin counts are allowed only for benchmark diagnostics |
| Engineering process | Write tests first before any system-specific implementation code |

Follow-up clarification after the first frozen `2D` pass:

- The current pair `multiwell_strong_transition` and `gated_local_linear` is still useful for transition and local-chart diagnostics, but it is not yet a complete benchmark suite for the stronger paper claim that periodic decode/re-encode helps by switching between attractor-neighborhood linearizations.
- That flagship follow-up is implemented locally as `gated_transfer_linear`, with the detailed mathematical specification and calibration policy in [chart_switching_transfer_system_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/chart_switching_transfer_system_plan_20260331.md).
- The lead open work after system construction is trained-model screening and claim selection, not more toy-system invention.

## Toy-system families

### Existing family to extend: multiwell bridge

Use the existing `multiwell_*` systems as the first deterministic family because they already support:

- native `2D` phase portraits,
- benchmark-only endpoint labeling via long rollout,
- compatibility with the current support-alignment and local-linearity tooling,
- a clean path to transition-rich behavior through deterministic vector-field modifications.

The first-pass roles are fixed as follows:

| System | Role | Basin count | Notes |
| --- | --- | --- | --- |
| `multiwell_gradient` | Low-transition control | `5` | Same attractor geometry, minimal transition pressure |
| `multiwell_strong_transition` | Main transition-rich positive candidate | `5` | Deterministic transition corridors already exist in the vector field |

Native `3D` variants are allowed later, but only after the `2D` systems are calibrated and selected.

### New family to add: gated local-linear system

Add one deterministic family with these properties:

- native `2D` dynamics at first pass,
- `3` attractors initially, scalable later,
- explicit attractor centers,
- explicit local Jacobians or local linear maps,
- explicit transition corridors or gate regions,
- long-rollout endpoint labels that do not require manual relabeling.

This family exists to support a deeper mechanics section in the paper. It should let us study:

- where local linearity holds,
- where transitions break local linearity,
- whether representation partitions align with known dynamical regions,
- whether forecasting failures come from within-regime prediction or transition handling.

The first-pass environment names are:

- `gated_local_linear`
- `gated_local_linear_3d`

### Flagship follow-up family: chart-switching transfer system

Add one follow-up deterministic family whose explicit purpose is to make chart switching between attractor neighborhoods visible and calibratable.

This family should:

- keep true long-run endpoint basins well defined,
- introduce non-invariant source neighborhoods around attractors,
- provide explicit exit sectors and transport channels between attractor neighborhoods,
- let us test whether periodic decode/re-encode helps because the correct local affine chart changes along the trajectory.

The detailed mathematical specification and implementation order live in [chart_switching_transfer_system_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/chart_switching_transfer_system_plan_20260331.md).

### More new families: TODO.

## Interfaces and required artifacts

### Environment names

- `multiwell_strong_transition_3d`
- `gated_local_linear`
- `gated_local_linear_3d`
- `gated_transfer_linear`
- `gated_transfer_linear_3d`

### Per-trajectory calibration fields

- `endpoint_basin`
- `region_path`
- `crossed_region`
- `crossing_count`
- `settling_window_basin`
- `endpoint_label_stable`
- `start_source_basin`
- `start_core_basin`
- `chart_switch_count`
- `channel_dwell_steps`
- `visited_channel`

### Per-system calibration summary fields

- `num_endpoint_basins`
- `endpoint_basin_distribution`
- `crossing_fraction_by_endpoint_basin`
- `overall_crossing_fraction`
- `label_stability_rate`
- `sparse_anchor_h1000_best_periodic`
- `source_neighborhood_count_by_basin`
- `source_neighborhood_transfer_fraction_by_basin`
- `overall_source_neighborhood_transfer_fraction`
- `core_count_by_basin`
- `core_retention_fraction_by_basin`
- `overall_core_retention_fraction`
- `chart_switch_count_distribution`
- `median_chart_switch_count_on_transfers`
- `channel_occupancy_fraction`
- `mean_channel_dwell_steps_on_transfers`

## Tests first

No system-specific environment code or metric code should be written until the test files below exist in `tests/`.

| Planned test module | What it must verify |
| --- | --- |
| `tests/test_transition_rich_system_determinism.py` | Fixed config plus fixed initial state always produce identical trajectories |
| `tests/test_transition_rich_endpoint_labels.py` | Endpoint labels are stable under longer rollout and do not change under a short horizon extension |
| `tests/test_transition_rich_crossing_metrics.py` | Region-path parsing, crossing counts, and crossing-fraction summaries are correct on hand-constructed paths |
| `tests/test_transition_rich_env_registry.py` | New environment names construct through the existing factory |
| `tests/test_transition_rich_calibration_schema.py` | Calibration summaries always emit the required per-trajectory and per-system fields |

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

Before any model training, screen candidate parameters offline from a fixed initial-condition box.

Each candidate must be evaluated on:

- endpoint-basin count,
- endpoint-basin occupancy,
- region paths,
- crossing fractions,
- label stability under longer rollout,
- sparse-anchor forecast sanity.

### Acceptance gates

| Gate | Requirement |
| --- | --- |
| Determinism | System is deterministic and reproducible from fixed config plus fixed initial state |
| Plotability | Native `2D` phase portrait is clean enough for paper figures |
| Basin count | Between `3` and `10` endpoint basins |
| Basin occupancy | No endpoint basin below `max(0.05, 0.5 / B)` occupancy, where `B` is the endpoint-basin count |
| Transition richness | Per-endpoint-basin crossing fraction in the acceptable `0.30-0.70` range for the frozen first-pass pair |
| Label stability | Endpoint labels remain stable under longer rollout |
| Forecast sanity | The sparse MLP anchor reaches the repo long-horizon sanity band: `H1000 best_periodic < 10` |

If strict per-endpoint-basin `0.30-0.70` cannot be met, reject the candidate unless:

- overall crossing remains high,
- no endpoint basin falls below `0.25` crossing fraction,
- and the system still has clean deterministic mechanics worth studying in depth.

### First-pass candidate freeze

Freeze exactly two paper candidates after calibration:

1. one transition-rich `multiwell` system,
2. one `gated_local_linear` system.

Both should be native `2D` in the first pass.

The explicit-transfer toy is a deliberate follow-up family with its own calibration rule: use source-neighborhood transfer fractions and inner-core retention rather than the endpoint-conditioned crossing gate.

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
- and report:
  - close-pair count,
  - branching-pair count,
  - branching rate,
  - branching severity,
  - region-conditioned breakdowns when region labels are available.

This diagnostic is not the headline metric, but it is important because the motivating systems are deterministic. Learned local partitions should not come at the cost of obvious forecast-side branching from effectively identical states.

## Experiment stages

### Stage 0: system calibration and freeze

Before any model sweep:

- implement tests,
- calibrate deterministic systems,
- save phase portraits and region overlays,
- freeze the paper-facing `2D` candidates,
- document calibration summaries in `docs/EXPERIMENTS.md` and `docs/PAPER_TRACK_STATUS.md`.

### Stage 1: trained-model screening

Run a small but decision-grade screening matrix on the frozen three-system suite:

- `multiwell_strong_transition`,
- `gated_local_linear`,
- `gated_transfer_linear`.

For each root:

- use the standard `200k` paper budget unless a narrower smoke pass is explicitly called out,
- keep seed coverage fixed,
- select checkpoints by the standard `evaluation_results_best.json` rule,
- report best-periodic and no-reencode comparisons.

The purpose of Stage 1 is not only to ask "does it forecast?" but to ask:

- which system is the clean mechanistic positive,
- which system is the harder stress test,
- whether the secondary toy is actually paper-relevant.

### Stage 2: chart-change attribution

If a system is meant to support a chart-switching causal claim, run a post hoc attribution pass that compares periodic gains on:

- chart-change windows,
- non-switch windows,
- transition regions,
- stable regions,
- transfer-start subsets when applicable.

If those gains do not localize where the true chart changes occur, do not make the stronger causal claim.

### Stage 3: support local-linearity

Run recurring-support local-linearity on the collected checkpoints to test whether learned supports define reusable local predictive structure.

Required controls:

- global local-dynamics fit,
- shuffled support assignment,
- matched zero-sparsity MLP where the question is whether explicit sparsity itself matters.

## Paper-facing interpretation rules

Use the branch to support the following scientific story:

1. Learned sparse supports may define meaningful dynamical partitions without training-time basin labels or known basin counts.
2. Those partitions should correspond to regions with reusable local predictive structure.
3. Forecast failures should be studied in the context of regime changes and stale local charts, not treated as unexplained MSE spikes.

Do **not** rely on the following shortcuts:

- do not assume the number of basins is known at training time,
- do not write the paper as if benchmark-only region labels were training supervision,
- do not treat basin-block alignment as the primary target unless a specific experiment requires it,
- do not claim a chart-change-localization win unless the attribution read is actually positive.

Prioritize **basin-support alignment**: each endpoint basin should map to a unique sparse support, even when trajectories pass through shared transitions.

## Immediate execution checklist

1. Keep the deterministic `2D` suite as the active lead branch for the paper.
2. Maintain the supporting benchmark, hard-system, and older mechanism packets as frozen context rather than as the live execution branch.
3. Keep `docs/EXPERIMENTS.md` and `docs/PAPER_TRACK_STATUS.md` aligned whenever calibration results or paper positioning changes.
4. When handing the branch to senior coauthors, describe experimental protocols and system mechanics clearly, without internal code names.
5. If the transfer toy supports only a weaker stress-test interpretation, write it that way rather than forcing a stronger story.

## Expected paper role split

The intended suite-level split is:

- `gated_local_linear`: clean mechanistic positive,
- `gated_transfer_linear`: explicit-transfer stress test,
- `multiwell_strong_transition`: weaker secondary corridor toy.

If later evidence contradicts that split, update this file, `docs/EXPERIMENTS.md`, and `docs/PAPER_TRACK_STATUS.md` together.
