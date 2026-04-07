# High-Dimensional Multi-Basin Benchmark Systems

**Goal**: Implement genuinely high-dimensional (>5D, scalable) dynamical systems with
multiple basins of attraction as benchmark environments for the SKAE project.

**Motivation**: All current multi-basin environments (Duffing, Lyapunov,
MultiWell, Blended) are fundamentally 2D with optional zero-padding to ~8D. This
limits our ability to test whether LISTA-based Koopman encoders can discover
sparse subspace structure in systems where high dimensionality is *intrinsic*
rather than artificial. These benchmarks will stress-test whether basin-support
alignment and union-of-subspaces structure emerge in realistic high-dimensional
settings.

---

## Candidate Systems (Priority Order)

### 1. Coupled Kuramoto Oscillators on a Ring

**Dimension**: N (one phase variable per oscillator; set N = 8, 16, 32, 64+)

**Dynamics**:
$$\dot{\theta}_i = \omega_i + \frac{K}{N} \sum_{j \in \mathcal{N}(i)} \sin(\theta_j - \theta_i), \quad i = 1, \dots, N$$

For a ring topology, $\mathcal{N}(i) = \{i-1, i+1\}$ (nearest-neighbor coupling).

**Attractors**: Multiple coexisting phase-locked states characterized by distinct
winding numbers $q$ (the total phase twist around the ring). For N oscillators
there are $O(N)$ stable twisted states. Chimera states (partial synchrony) also
appear for larger N.

**Rationale for SKAE**:
- Dimension is a *free parameter* — trivial to sweep from 8 to 100+.
- Each winding number $q$ defines a qualitatively different attractor → natural
  candidate for distinct sparse supports in the encoder's output.
- The ODE is simple (single sinusoidal coupling) → cheap to simulate.
- Well-studied basin geometry: basins are "octopus-shaped" in high dimensions
  (Basins Zoo, 2025), providing a rich test for generalization.
- Relevant to power grid synchronization (applied motivation).

**Configurable parameters**:
- `N`: number of oscillators (= state dimension)
- `K`: coupling strength (controls number of stable states)
- `topology`: `ring` (default), `all_to_all`, or custom adjacency
- `omega_mode`: `identical` (all ω=0), `uniform_spread`, or `random`
- `dt`: integration time step (default 0.05)

**Existing implementations**:
- [`fabridamicelli/kuramoto`](https://github.com/fabridamicelli/kuramoto) — numpy/scipy, pip-installable, supports arbitrary adjacency graphs. We would *not* depend on this; it's reference-only. The ODE is trivial to implement in PyTorch.
- [`icemtel/carpet`](https://github.com/icemtel/carpet) — coupled phase oscillators with numba acceleration.
- No external dependency needed; 15-25 lines of PyTorch in `dynamics_fn`.

**Basin ground-truth for evaluation**: Given a converged trajectory, the winding
number $q = \frac{1}{2\pi}\sum_i (\theta_{i+1}^* - \theta_i^*)$ (mod ring) is a
discrete integer label. This gives us a *natural* basin label without requiring
ground-truth knowledge of the ODE parameters — just run the trajectory to
convergence and read off $q$.

---

### 2. Continuous Hopfield Network

**Dimension**: N (one continuous neuron activation per unit; set N = 8, 16, 32, 64+)

**Dynamics** (Hopfield, 1984):
$$\tau \dot{x}_i = -x_i + \sum_{j=1}^{N} W_{ij} \tanh(\beta x_j) + b_i$$

where $W = \frac{1}{P}\sum_{\mu=1}^P \xi^\mu (\xi^\mu)^\top$ is the Hebbian weight
matrix storing $P$ memory patterns $\xi^\mu \in \{-1, +1\}^N$.

**Attractors**: Each stored pattern $\xi^\mu$ is a stable fixed point. Storing $P$
patterns → $P$ primary basins (plus $O(P^2)$ spurious mixture states at low
load $\alpha = P/N$). For $\alpha < 0.14$, retrieval is reliable; above this,
a spin-glass phase emerges.

**Rationale for SKAE**:
- *Dimension and basin count are independently controllable*: set $N$ for
  dimension, $P$ for number of basins.
- All attractors are *fixed points* (gradient flow on energy landscape) —
  forecasting is well-defined and there is no chaotic regime to confound evaluation.
- The sparse support hypothesis has a direct physical interpretation: each memory
  pattern activates a different subset of neurons → different patterns should map
  to different sparse supports in the LISTA encoding.
- Energy function $E = -\frac{1}{2}x^\top W x$ provides an independent
  convergence diagnostic.
- Classical, well-understood system — results are easy to communicate in a paper.

**Configurable parameters**:
- `N`: number of neurons (= state dimension)
- `P`: number of stored patterns (= number of basins)
- `beta`: nonlinearity gain (inverse temperature; default 1.0)
- `tau`: time constant (default 1.0)
- `pattern_mode`: `random` (iid ±1), `orthogonal`, or `structured`
- `dt`: integration time step (default 0.05)

**Existing implementations**:
- [`ptolmachev/Hopfield_Nets`](https://github.com/ptolmachev/Hopfield_Nets) — continuous Hopfield ODE in Python. Reference only.
- [`ml-jku/hopfield-layers`](https://github.com/ml-jku/hopfield-layers) — modern Hopfield layers in PyTorch (different focus but useful reference for the $\tanh$ update).
- No external dependency needed; ~20 lines of PyTorch. The weight matrix $W$ is
  precomputed from patterns at init time.

**Basin ground-truth for evaluation**: After convergence, compute overlap
$m^\mu = \frac{1}{N}\sum_i \xi^\mu_i \operatorname{sign}(x_i)$ for each pattern.
The pattern with highest overlap identifies the basin. This is a clean, integer label.

---

### 3. Competitive Lotka-Volterra (N species)

**Dimension**: N (one population variable per species; set N = 6, 10, 20+)

**Dynamics**:
$$\dot{x}_i = x_i \left( r_i - \sum_{j=1}^{N} A_{ij} x_j \right), \quad x_i \geq 0$$

where $r_i > 0$ is the intrinsic growth rate and $A$ is the interaction matrix
($A_{ii} = 1$ for self-limitation, $A_{ij} > 0$ for competition).

**Attractors**: Multiple stable equilibria, each corresponding to a different
subset of surviving species ($x_i > 0$). Smale (1976) proved that for $N \geq 5$,
the dynamics on the carrying simplex can exhibit *any* asymptotic behavior.
With carefully chosen $A$, the number of stable equilibria grows combinatorially.

**Rationale for SKAE**:
- Different equilibria literally correspond to different "active" species — a
  direct analogy to different sparse supports where different basis elements
  are active. This is the *most natural* system for testing the union-of-subspaces
  hypothesis.
- The state space has a natural positivity constraint ($x_i \geq 0$), which
  adds interesting structure vs. unbounded systems.
- Ecologically meaningful: each basin = a different community composition.
- Basin sizes scale roughly exponentially with total biomass at equilibrium
  (recent result, Springer 2025), providing quantitative predictions to verify.

**Configurable parameters**:
- `N`: number of species (= state dimension)
- `interaction_mode`: `symmetric` (random symmetric $A$), `asymmetric`,
  `block_diagonal` (communities), or `preset` (hand-tuned for known equilibria)
- `r_mode`: `uniform`, `heterogeneous`
- `dt`: integration time step (default 0.01, needs to be small for stiffness)
- `positivity_clip`: whether to enforce $x_i \geq 0$ after each step

**Existing implementations**:
- [`FMagnani/Generalized-Lotka-Volterra_N-species-model`](https://github.com/FMagnani/Generalized-Lotka-Volterra_N-species-model) — generic N-species scipy solver. Reference only.
- SciPy cookbook tutorial for 2-species. We generalize to N.
- No external dependency needed; ~15 lines of PyTorch.

**Basin ground-truth for evaluation**: At equilibrium, the *surviving species set*
$S = \{i : x_i > \epsilon\}$ is the basin label. This is a combinatorial label
(a subset of $\{1, \dots, N\}$) that maps naturally to a binary support vector.

**Implementation challenge**: The positivity constraint ($x_i \geq 0$) means
trajectories can hit the boundary of the positive orthant. Need to clip or use
a log-transform ($y_i = \log x_i$) to keep dynamics smooth.

---

### 4. Power Grid Swing Equations

**Dimension**: 2N (angle $\delta_i$ + frequency $\dot\delta_i$ per generator; N = 4, 8, 16+)

**Dynamics**:
$$M_i \ddot{\delta}_i + D_i \dot{\delta}_i = P_i - \sum_{j=1}^{N} B_{ij} \sin(\delta_i - \delta_j)$$

Second-order ODE → first-order system of dimension 2N:
$$\dot{\delta}_i = \omega_i, \quad \dot{\omega}_i = \frac{1}{M_i}\left(P_i - D_i \omega_i - \sum_j B_{ij}\sin(\delta_i - \delta_j)\right)$$

**Attractors**: Synchronous operating state(s) where all generators lock to a
common frequency. Multiple stable frequency-locked equilibria coexist; the
"desirable" basin is synchrony, while others correspond to partial
desynchronization or blackout states.

**Rationale for SKAE**:
- Physically meaningful application: power grid stability is a major engineering
  concern, and basin-of-attraction analysis is standard in the field.
- Second-order dynamics add richer structure than first-order gradient flows.
- Natural "control" motivation: the project's long-term goal is LQR control,
  and power grid stabilization is a canonical control problem.
- Basin stability is a well-studied metric → easy to compare against literature.

**Configurable parameters**:
- `N`: number of generators (state dim = 2N)
- `topology`: `ring`, `star`, `IEEE_9bus`, `IEEE_14bus`, `random_graph`
- `M_i`, `D_i`: inertia and damping (homogeneous or heterogeneous)
- `P_i`: power injections (controls operating point)
- `B_ij`: susceptance matrix (from topology)
- `dt`: integration time step (default 0.01)

**Existing implementations**:
- [`CURENT/andes`](https://github.com/CURENT/andes) — full-featured power system transient simulator. Too heavy as a dependency, but good reference for standard test cases (IEEE bus systems).
- The swing equation is simple enough to implement directly (~20 lines PyTorch).
- Standard IEEE test cases (9-bus, 14-bus, 39-bus) give ready-made topologies.

**Basin ground-truth for evaluation**: Check whether all generators converge to a
common frequency ($|\omega_i - \bar\omega| < \epsilon$ for all $i$). If yes →
"synchronous" basin. If some subset desynchronizes → label by the synchronized
cluster partition.

---

### 5. Gene Regulatory Network (Toggle Switches)

**Dimension**: 2N–3N (protein + mRNA per gene; N = 3–10 genes → 6–20D)

**Dynamics** (generalized toggle switch):
$$\dot{p}_i = \alpha_i \cdot \frac{K_i^n}{K_i^n + \left(\sum_{j \in \text{rep}(i)} p_j\right)^n} - \gamma_i p_i$$

where $p_i$ is protein concentration for gene $i$, $\text{rep}(i)$ is the set
of repressors of gene $i$, and $n$ is the Hill coefficient (cooperativity).

**Attractors**: Each stable equilibrium corresponds to a cell fate where certain
genes are "on" (high expression) and others are "off" (low expression). A
network of $N$ mutually repressing genes can have up to $N!$ stable states
(though typically far fewer).

**Rationale for SKAE**:
- Biologically motivated with direct real-world significance (cell
  differentiation, cellular decision-making).
- The on/off pattern of gene expression at each attractor maps naturally to a
  binary support pattern → ideal for testing sparse support alignment.
- Hill function nonlinearity is qualitatively different from sinusoidal
  (Kuramoto) or polynomial (Duffing) → tests generality of the approach.

**Configurable parameters**:
- `N`: number of genes (state dim = 2N with protein + mRNA, or N with protein only)
- `network_topology`: `ring_repression`, `all_pairs`, `random`, `toggle_3`
- `hill_n`: Hill coefficient (default 4; higher = sharper switches)
- `alpha`, `gamma`, `K`: production rate, degradation rate, half-activation
- `dt`: integration time step (default 0.05)

**Existing implementations**:
- No ready-made PyTorch library. Standard systems biology implementations use
  scipy. We implement from scratch in PyTorch (~25 lines).
- The repressilator (3-gene ring) is a well-known test case.

**Basin ground-truth for evaluation**: At equilibrium, binarize gene expression:
$b_i = \mathbf{1}[p_i > \text{threshold}]$. The binary vector $b$ is the basin label.

---

### 6. Coupled Chua Circuits

**Dimension**: 3N (3 state variables per circuit, N circuits; N = 2–6 → 6–18D)

**Dynamics** (single Chua circuit):
$$\dot{x}_i = \alpha(y_i - x_i - f(x_i)), \quad \dot{y}_i = x_i - y_i + z_i, \quad \dot{z}_i = -\beta y_i$$

where $f(x) = m_1 x + \frac{1}{2}(m_0 - m_1)(|x+1| - |x-1|)$ is the
piecewise-linear Chua diode.

Coupling: $\dot{x}_i \mathrel{+}= \epsilon \sum_{j \in \mathcal{N}(i)} (x_j - x_i)$

**Attractors**: Each isolated Chua circuit has a double-scroll attractor. When
coupled, the system exhibits multi-scroll attractors, hyperchaos (two positive
Lyapunov exponents for N ≥ 2), and coexisting attractors with distinct basins.

**Rationale for SKAE**:
- Already partially supported via dysts (single Chua). Extending to coupled
  circuits is a natural generalization.
- Chaotic + multi-basin → hardest test case. If LISTA can find structure here,
  it validates the approach beyond fixed-point attractors.
- The piecewise-linear nonlinearity is a good stress test for the encoder.

**Configurable parameters**:
- `N`: number of coupled circuits (state dim = 3N)
- `alpha`, `beta`, `m0`, `m1`: single-circuit parameters
- `epsilon`: coupling strength
- `topology`: `chain`, `ring`, `all_to_all`
- `dt`: integration time step (default 0.01)

**Existing implementations**:
- dysts already has `Chua` (3D single) and `MultiChua` — could use as base.
- Reference: [Multiscroll attractor (Wikipedia)](https://en.wikipedia.org/wiki/Multiscroll_attractor).
- ~25 lines of PyTorch for the coupled version.

**Basin ground-truth for evaluation**: Harder than the others since attractors are
chaotic, not fixed points. Would need to identify attractors by Lyapunov
exponent signature or by clustering long trajectories in a lower-dimensional
projection.

---

## Implementation Plan

### Phase 1: Infrastructure (shared across all systems)

1. **Add config dataclasses** in `skae/config.py`:
   - `KuramotoConfig`, `HopfieldConfig`, `CompetitiveLVConfig`,
     `SwingConfig`, `GeneRegConfig`, `CoupledChuaConfig`
   - Wire into `EnvConfig` and `EnvConfig.from_dict()`.

2. **Implement all six `Env` subclasses** in `skae/data.py`:
   - Follow existing pattern: `__init__(cfg)`, `reset(rng)`, `step(state)`.
   - All use `integrate_rk4` with a `dynamics_fn` closure.
   - Each class has a `basin_label(state)` method returning an integer basin
     ID for evaluation (not used during training).

3. **Register in `_ENV_REGISTRY`**:
   - `"kuramoto"`, `"hopfield"`, `"competitive_lv"`, `"swing"`,
     `"gene_reg"`, `"coupled_chua"`

### Phase 2: Core Three (implement first)

Implement **Kuramoto**, **Hopfield**, and **Competitive LV** first because:
- They cover the three most distinct attractor types: phase-locked (Kuramoto),
  fixed-point gradient (Hopfield), boundary-constrained (LV).
- Dimension and basin count are independently scalable in all three.
- Basin labeling is clean and unambiguous.
- No chaotic dynamics to complicate evaluation.

### Phase 3: Validation

For each system:
1. **Smoke test**: Generate 100 trajectories, verify they converge to distinct
   attractors, plot in 2D PCA projection.
2. **Basin labeling test**: Verify `basin_label()` returns consistent labels
   for trajectories initialized near known attractors.
3. **Dimension sweep**: Run with N = 8, 16, 32, 64 and verify reasonable
   integration time and memory usage.

### Phase 4: Extended Systems

Implement **Swing**, **Gene Regulatory**, and **Coupled Chua** as second-tier
benchmarks for the paper.

---

## Dependency Summary

| System | External deps | PyTorch LOC | Difficulty |
|--------|--------------|-------------|------------|
| Kuramoto | None | ~25 | Easy |
| Hopfield | None | ~30 | Easy |
| Competitive LV | None | ~25 | Easy (clip/log-transform) |
| Swing | None | ~30 | Medium (topology setup) |
| Gene Regulatory | None | ~30 | Medium (Hill functions) |
| Coupled Chua | None | ~35 | Medium (piecewise nonlinearity) |

**Total**: ~175 lines of new dynamics code + ~120 lines of config. Zero new
external dependencies.

---

## Integration with Existing Evaluation

The `basin_label()` method on each environment enables reuse of existing
evaluation metrics in `skae/evaluation.py`:
- **CosSep** (cosine separability between basin clusters in latent space)
- **Support overlap** (do different basins activate different sparse supports?)
- Basin-support alignment matrices (existing analysis pipeline)

No changes to the evaluation code are needed — just wire `basin_label()` into
the trajectory metadata during data collection.
