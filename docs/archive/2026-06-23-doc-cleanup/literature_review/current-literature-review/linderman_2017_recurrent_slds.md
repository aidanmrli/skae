# Linderman et al. (2017): Bayesian Learning and Inference in Recurrent Switching Linear Dynamical Systems

## Full citation

Scott W. Linderman, Matthew J. Johnson, Andrew C. Miller, Ryan P. Adams,
David M. Blei, and Liam Paninski. 2017. "Bayesian Learning and Inference in
Recurrent Switching Linear Dynamical Systems." In *Proceedings of the 20th
International Conference on Artificial Intelligence and Statistics*, PMLR
54:914-922.

## Sources read

Primary sources:

- PMLR proceedings page:
  <https://proceedings.mlr.press/v54/linderman17a.html>
- Main paper PDF:
  <https://proceedings.mlr.press/v54/linderman17a/linderman17a.pdf>
- Supplementary PDF:
  <https://proceedings.mlr.press/v54/linderman17a/linderman17a-supp.pdf>

No secondary summaries were used for the technical reading.

## Problem and motivation

The paper addresses time series with nonlinear dynamics that can be usefully
decomposed into simpler local dynamical regimes. Standard switching linear
dynamical systems (SLDSs) already represent this idea by combining a discrete
mode sequence with mode-specific linear-Gaussian dynamics. In a standard SLDS,
however, the discrete transition process is open loop:

```math
z_{t+1} \mid z_t, \{\pi_k\}_{k=1}^K \sim \pi_{z_t}.
```

That assumption is too restrictive when switching depends on the current
continuous state or on covariates. The motivating examples in the paper include
neuronal spiking, animal behavior, and basketball player trajectories, where a
switch can occur when the system enters a particular region of state space. A
standard Markov transition matrix cannot directly express "switch when the
continuous state crosses this region."

The goal is therefore to keep the interpretability and local linear structure
of SLDSs while allowing the discrete route to depend on the continuous latent
state. The key challenge is inference: once the discrete transition probability
depends nonlinearly on the continuous latent state, the conditional
linear-Gaussian structure used by Kalman-style message passing is broken.

## Method details

### Baseline SLDS structure

The paper starts from a conventional SLDS with a discrete latent mode
`z_t in {1, ..., K}`, a continuous latent state `x_t in R^M`, and observation
`y_t in R^N`. The dynamics are affine conditional on the mode:

```math
x_{t+1} = A_{z_{t+1}} x_t + b_{z_{t+1}} + v_t,
\qquad
v_t \sim \mathcal N(0, Q_{z_{t+1}}).
```

The observation model is linear-Gaussian:

```math
y_t = C_{z_t} x_t + d_{z_t} + w_t,
\qquad
w_t \sim \mathcal N(0, S_{z_t}).
```

In the experiments, the authors simplify by sharing `C`, `S`, and `d` across
modes. They place conjugate Dirichlet priors on transition rows and conjugate
matrix-normal inverse-Wishart priors on dynamics and emission parameters.

### Recurrent switching

The paper's recurrent SLDS replaces the open-loop transition matrix with a
state-dependent logistic transition model. The discrete state is generated as

```math
z_{t+1} \mid z_t, x_t, \{R_k,r_k\}_{k=1}^K
\sim \pi_{\rm SB}(\nu_{t+1}),
\qquad
\nu_{t+1} = R_{z_t} x_t + r_{z_t}.
```

Here `R_k` and `r_k` determine how the current continuous state affects the next
mode. The paper uses a stick-breaking logistic link rather than a standard
softmax. For `k < K`,

```math
\pi_{\rm SB}^{(k)}(\nu)
=
\sigma(\nu_k)\prod_{j<k}\sigma(-\nu_j),
```

and the remaining probability mass is assigned to state `K`:

```math
\pi_{\rm SB}^{(K)}(\nu)
=
\prod_{j=1}^{K-1}\sigma(-\nu_j).
```

The stick-breaking construction has a useful inference consequence: after
Polya-gamma augmentation, the recurrent transition factors become conditionally
Gaussian. Geometrically, the model partitions the continuous latent space by
linear hyperplanes, so different regions of state space can induce different
mode transition probabilities.

The paper also defines special cases:

- A recurrent AR-HMM, where the continuous state is observed directly.
- A shared-recurrence rSLDS, where `R` is shared across current modes.
- A recurrence-only variant, where `nu_{t+1} = R x_t + r`.
- A standard SLDS, recovered when the transition logits depend only on `z_t`.

### Polya-gamma augmentation and inference

The recurrent transition term introduces the non-Gaussian factor

```math
\psi(x_t,z_{t+1})
=
\prod_{k=1}^{K-1}
\sigma(\nu_{t+1,k})^{\mathbf 1[z_{t+1}=k]}
\sigma(-\nu_{t+1,k})^{\mathbf 1[z_{t+1}>k]}.
```

Equivalently,

```math
\psi(x_t,z_{t+1})
=
\prod_{k=1}^{K-1}
\frac{\exp(\nu_{t+1,k})^{\mathbf 1[z_{t+1}=k]}}
     {(1+\exp(\nu_{t+1,k}))^{\mathbf 1[z_{t+1}\ge k]}}.
```

The paper applies the Polya-gamma identity

```math
\frac{(e^\nu)^a}{(1+e^\nu)^b}
=
2^{-b}e^{\kappa\nu}
\int_0^\infty e^{-\omega\nu^2/2}p_{\rm PG}(\omega\mid b,0)\,d\omega,
\qquad
\kappa = a - b/2.
```

With auxiliary variables `omega_{t,k}`, the recurrent factor becomes
conditionally Gaussian in the logits:

```math
\psi(x_t,z_{t+1},\omega_t)
\propto
\prod_{k=1}^{K-1}
\exp\left\{
\kappa_{t+1,k}\nu_{t+1,k}
- \frac{1}{2}\omega_{t,k}\nu_{t+1,k}^2
\right\},
```

where

```math
\kappa_{t+1,k}
=
\mathbf 1[z_{t+1}=k]
- \frac{1}{2}\mathbf 1[z_{t+1}\ge k].
```

Since `nu_{t+1}` is linear in `x_t`, this restores Gaussian potentials over
the continuous state. The authors can then use standard message passing for
continuous states, HMM-style updates for discrete states, and conjugate updates
for dynamics, observations, and recurrence weights. The supplement gives a
structured mean-field and stochastic variational inference version, using
factors of the form

```math
q(z_{1:T})q(x_{1:T})q(\omega_{1:T})q(\theta).
```

The supplement also stresses initialization. The reported initialization
pipeline uses probabilistic PCA or factor analysis for continuous latents,
then an AR-HMM for discrete states and dynamics, and finally a greedy
decision-list/logistic-regression procedure to choose a stick-breaking order.
This is important because the stick-breaking likelihood depends on the ordering
of the discrete states.

## Results and limitations

### Main empirical results

The paper is primarily a methodological inference paper, with experiments that
illustrate the model rather than a large benchmark study.

1. Synthetic NASCAR example. The true system has four discrete states and a
   two-dimensional continuous latent trajectory tracing oval paths. Observations
   are ten-dimensional linear projections with Gaussian noise. The authors fit
   an rSLDS to `T = 10^4` time steps using `10^3` Gibbs iterations. The model
   recovers the four states and the oval continuous trajectory up to an affine
   non-identifiability. A standard SLDS can decode states, but the rSLDS is a
   better generative model because its switching depends on position.

2. Lorenz and Bernoulli-Lorenz examples. The rSLDS approximates the Lorenz
   attractor with two approximately linear rotational modes. The Bernoulli
   observation experiment uses 100-dimensional binary observations from a
   logistic GLM and holds out a time window. The model can infer switching
   uncertainty through the missing interval. In a posterior predictive check,
   the true maximum duration in the high-probability side is `215` time steps;
   an SLDS gives approximately `91 +/- 33`, while the rSLDS gives approximately
   `192 +/- 84`.

3. Basketball player trajectories. The authors fit a recurrence-only rAR-HMM
   with `K = 30` states to five Miami Heat player trajectories from a November
   1, 2013 game, totaling 256,103 time steps sampled every 40 ms. The inferred
   states are interpretable court-location-dependent behaviors. Held-out log
   likelihood improves slightly over an AR-HMM: `8.124` versus `8.110`
   nats/time step, with a random-walk baseline at `5.073`.

### Limitations

- The number of modes `K` is fixed in advance. This is a central difference
  from methods that try to avoid prespecifying a route or basin count.
- The route variable is an explicit latent discrete state in the model. This is
  powerful and interpretable, but it means the method is not discovering a
  route from an independently learned representation after training.
- Stick-breaking introduces ordering sensitivity. The supplement addresses
  this with a greedy decision-list initialization, but the dependence is still
  structural.
- Inference is approximate and initialization-sensitive. The paper uses Gibbs
  sampling and variational approximations, and the supplement explicitly notes
  the need for reasonable initialization.
- The continuous latent state is not identifiable except up to transformations,
  as the synthetic and Lorenz examples acknowledge.
- The observation and dynamics models are parametric and relatively structured:
  affine dynamics, linear-Gaussian or GLM-style observations, and logistic
  state-dependent switching.
- The empirical results are convincing demonstrations of mechanism, not broad
  evidence that rSLDS dominates alternatives across modern high-dimensional
  nonlinear forecasting benchmarks.

## Similarities to the current SKAE staged local `K_c` method

The relevant SKAE method has three stages:

1. Train one global sparse Koopman autoencoder with shared encoder, shared
   decoder, and one global latent transition matrix `K`.
2. Derive `C_stab` routes from learned sparse support-flow fates, without basin
   labels, basin counts, or trajectory-to-basin assignments.
3. Freeze the representation and train local affine latent maps

```math
T_c(z) = d_c + K_c(z-\bar z_c).
```

The similarities to Linderman et al. are substantial enough that this paper
should be treated as important adjacent prior art.

### Shared local-linear view of nonlinear dynamics

Both methods model globally nonlinear behavior by composing simpler local
linear or affine dynamics. In rSLDS, each discrete state selects an affine
linear dynamical system:

```math
x_{t+1}=A_{z_{t+1}}x_t+b_{z_{t+1}}+v_t.
```

In staged SKAE, each support-stable component selects a latent affine map:

```math
T_c(z)=d_c+K_c(z-\bar z_c).
```

Thus, both methods instantiate the same broad modeling principle: a nonlinear
system can be approximated by switching among simpler affine predictors.

### State-dependent routing

Both methods make the route depend on the current state rather than using only
an exogenous sequence or a fixed open-loop Markov chain.

In rSLDS, the route is probabilistic:

```math
z_{t+1}\sim \pi_{\rm SB}(R_{z_t}x_t+r_{z_t}).
```

In SKAE, the route is deterministic or fallback-based after support assignment:
the current latent state determines an active-coordinate support mask, that mask
maps to a base support-family label, and the base label maps to a support-flow
fate `c = C_stab(u_t)`. Forecasting recomputes the route from the model's own
current prediction, especially under periodic re-encoding.

### Discrete regime interpretability

Both methods expose a discrete object that can be inspected. For rSLDS, the
object is the posterior mode sequence and the learned region-dependent
transition probabilities. For SKAE, the object is the sparse support and its
support-flow fate. In both cases, the discrete object can be interpreted as a
regime variable, though the interpretation is different: rSLDS modes are
explicit latent states; SKAE support fates are emergent active-coordinate
objects.

### No direct regime-label supervision

Neither method requires observed mode labels during fitting. Linderman et al.
infer modes from time series under a generative model. SKAE derives `C_stab`
from supports produced by a globally trained model and does not use basin labels
or basin counts to form the route.

### Operational use of the route

Both routes are operational, not just diagnostic. In rSLDS, the inferred or
sampled mode determines the transition dynamics. In staged SKAE, `C_stab`
selects the local affine latent map used during rollout. This matters for our
paper because it places SKAE in the same broad family of state-dependent
local-linear dynamical models, not merely in the family of post-hoc clustering
diagnostics.

## Detailed differences from SKAE staged local `K_c`

### Route construction

The main difference is how the route is obtained.

In rSLDS, the route is an explicit model variable `z_t` from the beginning. The
model assumes a finite set of `K` discrete states and learns a parametric
transition law for them. The route exists as part of the generative story.

In staged SKAE, the first-stage model has no explicit route variable. It trains
one shared encoder, decoder, and global `K`. Only after this global sparse KAE
has learned a representation do we inspect the sparse support trajectories,
construct a support-transition graph, identify recurrent support-flow
components, and assign route labels by support-flow fate. The route is
therefore an emergent object of the learned sparse representation rather than a
latent variable assumed by the model.

### Assumptions about number of regimes

rSLDS requires the number of discrete states `K` to be fixed before inference.
The basketball example uses `K = 30`; the synthetic examples use known small
values such as four or two states. The method can be extended with Bayesian
nonparametrics in related work, but this paper's model is finite-`K`.

The SKAE staged method, as framed for our paper, does not know the number of
basins or support-stable route components in advance. `C_stab` is derived from
the empirical support-flow graph. The route count is an output of the support
construction, not an input mode cardinality. For benchmark evaluation, basin
labels and counts can be used after the fact to score alignment, but they are
not used to build `C_stab` or train `K_c`.

### Supervision and deployment setting

rSLDS is unsupervised with respect to mode labels, but it is supervised by a
modeling assumption: there is a discrete latent state sequence of size `K`, and
each mode owns an affine dynamics model. The inference procedure is built
around recovering that sequence.

SKAE's intended training and deployment setting is stricter with respect to
basins: the method is not told the number of basins, which trajectory belongs
to which basin, or which route should be active. The only training signal is
the autoencoding/forecasting/sparsity objective in stage one and the routed
forecasting objective in stage two. Labels are reserved for evaluation on
benchmarks.

### Representation learning

rSLDS uses a continuous latent state with structured probabilistic emissions.
The observations are generated from `x_t` through linear-Gaussian or GLM-style
observation models. The method is a Bayesian state-space model.

SKAE uses a neural encoder and decoder to learn a latent representation. The
sparse active-coordinate pattern is a central object. This is not merely a
dense latent state with an attached mode; the active support itself becomes the
candidate regime variable. The sparse support is both an interpretability
object and the basis for route construction.

### Training schedule

rSLDS learns modes, continuous states, recurrence weights, dynamics, and
emissions jointly under one probabilistic model. The inference algorithm
alternates updates over latent states, auxiliary variables, and parameters, but
the model itself is not staged in the SKAE sense.

SKAE is deliberately staged:

1. A single global sparse Koopman autoencoder is trained first.
2. The representation is frozen.
3. `C_stab` is built from support-flow fates.
4. Local affine maps `K_c,d_c` are trained while the encoder, decoder, and
   global `K` remain fixed.

This staging is methodologically important. It tests whether the sparse
representation learned under a global Koopman objective already contains a
useful basin-scale support object that can later be reused for local dynamics.

### Routing law: learned probabilistic classifier versus support-flow quotient

rSLDS routing is a learned probabilistic classifier over modes, using
stick-breaking logistic regression. Its decision boundaries are linear in the
continuous latent state, conditional on the current mode.

`C_stab` routing is not a classifier trained against mode assignments. It is a
quotient of support dynamics: exact sparse masks are grouped into base support
families, support-family transitions define a graph, recurrent components and
absorption fates define stable route labels, and those route labels select
local affine maps. This makes the SKAE route closer to a learned dynamical
invariant of support trajectories than to a parametric logistic gate.

### Treatment of uncertainty

rSLDS is Bayesian. It represents uncertainty over continuous states, discrete
states, dynamics, emissions, recurrence weights, and auxiliary variables. This
is central to the paper's claims about missing data and posterior predictive
checks.

The current SKAE staged local `K_c` method is not Bayesian. It uses point
estimates from neural training and deterministic support construction with
fallback rules. Its evidence is therefore about learned sparse representations
and forecasting performance, not posterior uncertainty or calibrated state
inference.

### Connection to Koopman learning

rSLDS is not a Koopman autoencoder. It is a switching state-space model with
local affine dynamics. It does not train a shared encoder/decoder around a
single global Koopman matrix and then derive support routes from that learned
representation.

SKAE explicitly starts from the Koopman-autoencoder premise: one latent
transition `K` is trained globally, then support-stable routes are used to relax
the single-global-map constraint. The local maps therefore arise as a second
stage on top of a learned sparse Koopman representation.

### What counts as the regime object

In rSLDS, the mode `z_t` is the regime. In SKAE, the primary object is
basin-support alignment: each basin should map to a unique sparse support or
support-flow fate in latent `z`. We should therefore avoid describing SKAE as
merely "another switching model." The route is a sparse support object produced
by representation learning, not a standalone hidden Markov mode.

## Novelty implications for our paper

Linderman et al. is strong prior art for several broad ideas:

- Switching among local affine dynamics.
- State-dependent switching rather than open-loop Markov switching.
- Learning discrete dynamical regimes without observed mode labels.
- Using a discrete route to obtain interpretable nonlinear dynamics from
  simpler linear pieces.

Our paper should not claim novelty for any of those broad concepts. In
particular, it would be too strong to write that our contribution is simply
"learning local linear models without labels" or "using state-dependent routing
for affine latent dynamics." rSLDS already does that in a Bayesian state-space
form.

The defensible novelty is more specific:

1. We first train a single global sparse Koopman autoencoder with shared
   encoder, shared decoder, and one global `K`; we do not begin with an
   explicit discrete switching model.
2. The route is derived after global training from learned sparse
   support-flow fates, not from a prescribed finite mode set or a parametric
   logistic transition model.
3. The method does not require basin labels, basin counts, or
   trajectory-to-basin assignments during training or deployment.
4. The sparse support itself is the interpretable regime object. This ties the
   local-linear route to active-coordinate identities in the learned Koopman
   representation.
5. The second stage freezes the representation and trains local affine latent
   maps `T_c(z)=d_c+K_c(z-\bar z_c)`, testing whether the discovered support
   object can operationally select basin-local dynamics.

The clean positioning is therefore:

> Linderman et al. show that recurrent switching state-space models can learn
> state-dependent switching among affine dynamical systems with efficient
> Bayesian inference. Our method addresses a different route-discovery problem:
> starting from a single globally trained sparse Koopman autoencoder, can the
> model's own sparse support-flow fates become a label-free basin-scale routing
> object for local affine latent dynamics?

This distinction preserves the novelty of the SKAE staged local `K_c` result
without overstating it. The right claim is not that SKAE invents switching
linear dynamics. The claim is that sparse Koopman representation learning can
produce an inspectable support-flow route, without known basin counts or labels,
and that this route can be reused to specialize local affine predictors while
keeping the learned encoder and decoder fixed.
