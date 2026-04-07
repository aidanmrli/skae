# Chart-Switching Transfer System Plan

_Detailed toy-system design plan for the transition-rich paper branch, March 31, 2026_

_Restored on April 6, 2026 from the surviving branch docs after an accidental local deletion. This copy is consistent with the implemented system and current paper-facing interpretation, but may not be byte-identical to the lost draft._

---

## Objective

Add one deterministic native-plot `2D` benchmark system whose main paper role is to make **chart switching** explicit:

- the system should have multiple stable endpoint attractors,
- each attractor neighborhood should admit its own local linear dynamics,
- trajectories should pass through explicit transfer regions where the active local linearization changes,
- periodic decode/re-encode should be interpretable as reselecting the appropriate local affine chart.

This system is meant to complement the broader transition-rich branch by making the local-chart story legible in geometry rather than only in metrics.

## Why a new system is needed

The current frozen pair is useful but incomplete:

- `gated_local_linear` is a strong exact-local-Jacobian toy, but it does not produce meaningful transfer from one attractor neighborhood to another.
- `multiwell_strong_transition` has visible shared corridors, but its transition metric is endpoint-conditioned and does not isolate the chart-switching phenomenon as cleanly as desired for the strongest paper story.

The new system should therefore separate three objects that had previously been blurred together:

- **endpoint basin**: the true long-run attractor assignment,
- **source neighborhood**: a non-invariant neighborhood around an attractor from which trajectories may either stay or transfer,
- **chart**: a local affine regime used for mechanistic interpretation.

## Scientific role in the paper

The system should support the following argument:

1. A nonlinear deterministic system with multiple endpoint attractors does not admit one globally valid state-level linear chart.
2. It can still admit several interpretable local affine charts.
3. Forecasting should work best when the model re-anchors its latent representation as trajectories move between those charts.
4. Periodic decode/re-encode can then be read as a model-side chart-switching mechanism.

Current paper-facing caveat:

- the implemented benchmark is still scientifically useful even if the strongest chart-localization claim does not hold on the collected learned-model pass,
- so the benchmark should be written as an explicit-transfer stress test unless stronger localization evidence is collected later.

## Proposed environment family

Reserve the environment names:

- `gated_transfer_linear`
- `gated_transfer_linear_3d`

The first pass should implement only the native `2D` system.

## Implementation status

As of April 1, 2026:

- `gated_transfer_linear` is implemented locally and wired through the environment factory.
- The calibrated default `2D` system clears the intended transfer-family acceptance gates on the fixed `17x17` screening grid:
  - source-neighborhood transfer fractions `0.333 / 0.444 / 0.333`
  - overall source-neighborhood transfer `0.370`
  - core-retention fractions `1.000 / 1.000 / 1.000`
  - overall core retention `1.000`
  - label stability `1.000`
- The paper-facing mechanics figures live under [docs/figures/chart_switching_transfer_20260331](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331).
- `gated_transfer_linear_3d` remains reserved only; the implemented benchmark is the native `2D` system.

## Mathematical definition

### State space

Let the state be

\[
x = (x_1, x_2) \in \mathbb{R}^2.
\]

The system is autonomous and deterministic:

\[
\dot{x} = f(x).
\]

### Attractor centers

Use `B = 3` endpoint attractors arranged on an equilateral triangle:

\[
\theta_i = \phi_0 + \frac{2\pi i}{3}, \qquad
c_i = R
\begin{bmatrix}
\cos \theta_i \\
\sin \theta_i
\end{bmatrix},
\qquad i \in \{0,1,2\}.
\]

Recommended first-pass scale:

- `R \in [1.6, 2.0]`
- calibrated default: `R = 1.85`, `\phi_0 = 0`

For each ordered pair `(i, j)`, define the source-side exit direction

\[
d^{exit}_{ij} = \frac{c_j - c_i}{\|c_j - c_i\|},
\qquad
m^{exit}_{ij} =
\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
d^{exit}_{ij}.
\]

The exit-sector geometry is defined with `d^{exit}_{ij}`, but the downstream channel lane is allowed to bend and therefore uses its own direction defined from the entry and handoff points below.

### Region decomposition

For each attractor `i`, define three kinds of local regions.

#### 1. Inner core

\[
C_i = \{x : \|x - c_i\| \le r_c\}.
\]

This is the only region that should behave like a true strongly stable local neighborhood of endpoint basin `i`.

#### 2. Transfer shell

Define an outer source radius `r_s > r_c` and a shell

\[
S_i = \{x : r_c < \|x - c_i\| \le r_s\}.
\]

Split this shell into:

- a **return shell** `S_{i \to i}`,
- one **exit sector** `S_{i \to j}` for each destination `j \neq i`.

For each destination `j`, define the exit sector by a half-angle `\alpha_{exit}`:

\[
S_{i \to j}
=
\left\{
x \in S_i :
\angle(x-c_i, d^{exit}_{ij}) \le \alpha_{exit}
\right\}.
\]

Then define the return shell as the remaining shell mass:

\[
S_{i \to i}
=
S_i \setminus \bigcup_{j \neq i} S_{i \to j}.
\]

The **source neighborhood** for attractor `i` is

\[
N_i^{source} = C_i \cup S_{i \to i} \cup \bigcup_{j \neq i} S_{i \to j}.
\]

This source neighborhood is intentionally not forward invariant.

#### 3. Transfer channels

For each ordered pair `(i, j)`, define a channel entrance point

\[
\sigma_{ij} = \operatorname{sign}(\sin(\theta_i - \theta_j)),
\qquad
e_{ij} = c_i + r_s d^{exit}_{ij} + \sigma_{ij}\,\ell\, m^{exit}_{ij},
\]

where `\ell` is the lane offset. Define the destination outward and tangent directions by

\[
u_j = \frac{c_j}{\|c_j\|},
\qquad
v_j =
\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
u_j.
\]

Then define the destination handoff point by

\[
q_{ij} = c_j + r_h u_j + \sigma_{ij}\,\ell\, v_j,
\]

where `r_h` is a destination handoff radius satisfying `r_c < r_h < R`.

The actual channel lane is then defined by

\[
d^{chan}_{ij} = \frac{q_{ij} - e_{ij}}{\|q_{ij} - e_{ij}\|},
\qquad
m^{chan}_{ij} =
\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
d^{chan}_{ij}.
\]

Let

\[
L_{ij} = (d^{chan}_{ij})^\top (q_{ij} - e_{ij}),
\qquad
s_{ij}(x) = (d^{chan}_{ij})^\top (x - e_{ij}),
\qquad
r_{ij}(x) = (m^{chan}_{ij})^\top (x - e_{ij}).
\]

Then define the channel tube

\[
H_{ij}
=
\{x : 0 \le s_{ij}(x) \le L_{ij},\ |r_{ij}(x)| \le w\},
\]

with channel half-width `w`.

## Local dynamics

The vector field is piecewise-affine by region.

### Core dynamics

Inside each core, use a stable affine system centered on `c_i`:

\[
\dot{x} = A_i (x - c_i),
\qquad x \in C_i.
\]

Each `A_i` must be Hurwitz and distinct across basins. Use different local eigenstructures so that the charts are not interchangeable.

Recommended first-pass templates:

\[
\bar{A}_0 =
\begin{bmatrix}
-1.0 & -1.1 \\
1.1 & -1.0
\end{bmatrix},
\quad
\bar{A}_1 =
\begin{bmatrix}
-1.4 & 0.2 \\
-0.2 & -0.7
\end{bmatrix},
\quad
\bar{A}_2 =
\begin{bmatrix}
-0.8 & -0.3 \\
0.5 & -1.3
\end{bmatrix}.
\]

Optionally rotate each template by the basin angle to keep the geometry aligned with the triangle.

### Return-shell dynamics

In the return shell, keep a stable affine map back toward the same attractor:

\[
\dot{x} = \tilde{A}_i (x - c_i),
\qquad x \in S_{i \to i},
\]

with `\tilde{A}_i` Hurwitz and typically weaker than `A_i`.

This region should produce trajectories that stay with endpoint basin `i`.

### Exit-shell dynamics

For each exit shell `S_{i \to j}`, use a directed transport field with constant forward motion plus transverse contraction back toward the centerline:

\[
\dot{x}
=
\nu^{exit}_{ij} d^{exit}_{ij}
- \kappa^{exit}_{ij} m^{exit}_{ij} (m^{exit}_{ij})^\top (x - c_i),
\qquad x \in S_{i \to j},
\]

with `\nu^{exit}_{ij} > 0` and `\kappa^{exit}_{ij} > 0`.

This is still affine in `x`, but it matches the implemented mechanics more directly than a stable equilibrium beyond the shell approximation.

### Channel dynamics

Inside the channel, use constant forward drift with transverse contraction:

\[
\dot{s}_{ij} = \nu_{ij},
\qquad
\dot{r}_{ij} = -\kappa_{ij} r_{ij},
\qquad x \in H_{ij},
\]

with `\nu_{ij} > 0` and `\kappa_{ij} > 0`.

In Cartesian coordinates this becomes

\[
\dot{x}
=
\nu_{ij} d^{chan}_{ij}
- \kappa_{ij} m^{chan}_{ij} (m^{chan}_{ij})^\top (x - e_{ij}),
\qquad x \in H_{ij}.
\]

This is affine, deterministic, visually interpretable, and makes the active chart change obvious.

### Background return dynamics

Any state not in a core, exit shell, or channel is assigned to its nearest attractor center:

\[
i^*(x) = \arg\min_i \|x - c_i\|,
\]

and evolves under a return map

\[
\dot{x} = \hat{A}_{i^*(x)} (x - c_{i^*(x)}).
\]

This ensures trajectories not captured by a transfer region still settle into a well-defined endpoint basin.

### Smoothness policy

First pass:

- implement the system as piecewise-affine with deterministic hard region boundaries,
- use sufficiently small `dt`,
- accept boundary non-smoothness if the phase portrait and calibration are numerically stable.

Second pass only if needed:

- add thin blending strips between adjacent regions,
- interpolate affine fields with scalar gate functions.

The first pass should favor interpretability and exact local charts over global smoothness.

## Region labels and basin labels

The environment should expose both:

- `region_label(x)`: exact chart or region id,
- `basin_label(x)`: intended long-run endpoint basin id used for benchmark evaluation.

Recommended `region_label` structure:

- `core_i`
- `return_i`
- `exit_{i->j}`
- `channel_{i->j}`

Treat:

- `basin_label` as the endpoint evaluation label,
- `region_label` as the mechanistic chart label,
- `source_neighborhood_label` as a start-set diagnostic, not as the training target.

## Calibration quantities

The explicit-transfer toy should no longer use the old endpoint-conditioned crossing fraction as its main acceptance gate.

Instead, calibrate:

### Endpoint-basin stability

- endpoint-basin count,
- endpoint-basin occupancy,
- long-rollout label stability.

### Source-neighborhood transfer

For each source neighborhood `N_i^{source}`, compute:

- number of starts in that source neighborhood,
- fraction that end in basin `i`,
- fraction that transfer to each destination basin,
- total transfer fraction out of source basin `i`.

Summarize:

- `source_neighborhood_count_by_basin`,
- `source_neighborhood_transfer_fraction_by_basin`,
- `overall_source_neighborhood_transfer_fraction`.

### Core retention

For starts inside each inner core `C_i`, compute:

- count of core starts,
- fraction that remain with endpoint basin `i`.

Summarize:

- `core_count_by_basin`,
- `core_retention_fraction_by_basin`,
- `overall_core_retention_fraction`.

### Chart-switch and channel statistics

Also report:

- chart-switch count distribution,
- median chart-switch count on transfer trajectories,
- channel occupancy fraction,
- mean channel dwell steps on transfer trajectories.

### Forecast sanity

Before spending large model budget, require the sparse MLP anchor to reach the standard long-horizon sanity band on the calibrated default.

## Acceptance gates

| Gate | Requirement |
| --- | --- |
| Determinism | Fixed config plus fixed initial state must produce identical trajectories |
| Plotability | Region map, flow field, and trajectory views must be paper-usable in native `2D` |
| Endpoint basins | Exactly `3` endpoint basins on the default system |
| Occupancy | No endpoint basin should be degenerate on the fixed screening grid |
| Source-neighborhood transfer | Each source basin should have nonzero transfer, with overall transfer roughly in the intended intermediate band |
| Core retention | Core retention should be near-perfect; default target `>= 0.95`, calibrated result `1.000` |
| Label stability | Endpoint labels should be stable under longer rollout |
| Interpretability | Exit sectors and channels should be visually obvious in state space |

The calibrated default currently satisfies these with:

- transfer fractions `0.333 / 0.444 / 0.333`,
- overall transfer `0.370`,
- core retention `1.000`,
- label stability `1.000`.

## Required tests

The implementation should be covered by tests that isolate the benchmark semantics, not just the numerics.

Required system-specific tests:

- `tests/test_chart_switching_transfer_regions.py`
  - region ids are well formed and mutually exclusive,
  - representative points fall into the intended `core`, `return`, `exit`, and `channel` regions.
- `tests/test_chart_switching_transfer_channels.py`
  - channel membership behaves correctly across entry, lane center, and outside-channel cases.
- `tests/test_chart_switching_transfer_endpoint_labels.py`
  - endpoint labels are stable under longer rollout,
  - exit and channel regions map to the intended destination basin.
- `tests/test_chart_switching_transfer_system_determinism.py`
  - repeated rollout with the same configuration and start state is deterministic.

The generic transition-rich suite should also cover:

- environment registration,
- calibration schema,
- crossing and path parsing where applicable.

## Implementation order

1. Write the tests and verify failure.
2. Add the minimal environment scaffolding and label interfaces.
3. Implement the region decomposition.
4. Implement the piecewise-affine dynamics.
5. Add calibration utilities and summaries.
6. Generate ground-truth figures before queueing model training.
7. Only then run trained-model screening and mechanistic diagnostics.

## Figure packet

The benchmark should ship with a small deterministic figure packet that makes the mechanics visible:

- region map,
- chart-colored trajectories,
- endpoint-colored trajectories,
- uniform-grid starts,
- source-neighborhood starts,
- flow field,
- transfer summary.

The current figure set lives under [docs/figures/chart_switching_transfer_20260331](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331).

## Paper-facing interpretation rules

If the benchmark behaves as intended in geometry but the learned-model attribution read is weak, write it as a **hard explicit-transfer stress test**, not as the sole flagship positive.

Current branch interpretation:

- the system succeeds as a deterministic explicit-transfer benchmark,
- periodic re-encoding helps overall on the trained-model read,
- but the stronger claim that gains localize at true chart-change windows is not currently supported by the collected attribution pass.

Therefore:

- keep the benchmark in the paper,
- use it to show the hard regime where stale local charts matter,
- do not make it carry the cleanest chart-switching causal claim on its own.

## Recommended role split within the three-system suite

- `gated_local_linear`: clean mechanistic positive and cleanest chart-switching benchmark in state space,
- `gated_transfer_linear`: explicit-transfer stress test with transfer lanes and non-invariant source neighborhoods,
- `multiwell_strong_transition`: weaker shared-corridor transition toy.

If a future cleaner localization benchmark is added, revisit this split explicitly rather than silently upgrading the claim carried by `gated_transfer_linear`.
