# Literature Review Note: Takeishi et al. (2018), Factorially Switching DMD

## Full citation

Naoya Takeishi, Takehisa Yairi, and Yoshinobu Kawahara. 2018.
"Factorially Switching Dynamic Mode Decomposition for Koopman Analysis of
Time-Variant Systems." In *2018 IEEE Conference on Decision and Control
(CDC)*, 6402-6408. IEEE. DOI:
[10.1109/CDC.2018.8619846](https://doi.org/10.1109/CDC.2018.8619846).

## Source links and reading provenance

- Main article text read from the author-hosted PDF:
  <https://ntake.jp/paper/cdc2018_takeishi_paper.pdf>
- Bibliographic metadata checked against the authors' lab publication page:
  <https://ailab.t.u-tokyo.ac.jp/archives/wp-publications/takeishifactoriallyswitchingdynamic2018>
- DOI landing link:
  <https://doi.org/10.1109/CDC.2018.8619846>

I used the paper itself and primary publication metadata only. I did not rely on
secondary summaries.

## Problem and motivation

The paper addresses a limitation of standard dynamic mode decomposition (DMD)
and many DMD variants: they assume that the target dynamical system is
time-invariant over the analyzed data. This assumption is problematic when
system parameters change because of unobserved external effects, when data
contain transient behavior between unstable equilibria and attractors, or when
temporally local phenomena rise and fall suddenly.

The authors frame the problem in Koopman terms. For a time-varying discrete
dynamical system

$$
v_{i+1} = f_i(v_i),
$$

the Koopman operator at time index \(i\) is

$$
K_i g(v) = g(f_i(v)).
$$

If one applies an ordinary DMD model to data whose effective dynamics change
over time, the computed modes and eigenvalues may mix global, transient, and
localized effects into a single time-invariant approximation. The paper's
motivating idea is that even when the system varies over time, some spectral
components may be reused. Time variation can then be represented by switching
individual dynamic modes on and off, rather than by fitting a single global DMD
model or manually segmenting the trajectory.

## Method details

The proposed method is Factorially Switching Dynamic Mode Decomposition
(FSDMD). It extends probabilistic DMD by placing a binary activation variable on
each dynamic mode at each time. The word "factorial" refers to the fact that the
overall switching state is the product of many per-mode on/off variables, rather
than a single categorical regime variable.

### Koopman/DMD background used by the paper

For snapshot pairs

$$
(x_i,y_i) = (g(v_i), g(v_{i+1})),
$$

with \(g\) an \(m\)-dimensional observable, the usual finite-dimensional
Koopman/DMD picture assumes that the observables span an invariant subspace.
Then the snapshots can be decomposed as

$$
x_i = \sum_{j=1}^r \phi_j(v_i) w_j,
\qquad
y_i = \sum_{j=1}^r \lambda_j \phi_j(v_i) w_j,
$$

where \(\phi_j\) are Koopman eigenfunctions, \(\lambda_j\) are eigenvalues, and
\(w_j\) are Koopman modes, called dynamic modes in the DMD context.

Standard DMD effectively estimates such a decomposition under a time-invariant
assumption. FSDMD keeps the modal decomposition structure but lets each mode
contribute only when its binary switch is active.

### Observation model

For snapshot pair \(i\), FSDMD introduces latent coefficient vectors
\(\chi_i,\psi_i \in \mathbb{C}^r\). The observation likelihood is

$$
p(x_i,y_i \mid \chi_i,\psi_i)
= p(x_i \mid \chi_i)p(y_i \mid \psi_i),
$$

with

$$
x_i \mid \chi_i \sim \mathcal{CN}(W\chi_i,\sigma^2 I),
$$

$$
y_i \mid \psi_i \sim \mathcal{CN}(W\Lambda\psi_i,\sigma^2 I).
$$

Here \(W \in \mathbb{C}^{m \times r}\) contains the dynamic modes,
\(\Lambda\in\mathbb{C}^{r \times r}\) is diagonal with
\(\lambda=\operatorname{diag}(\Lambda)\), and \(\sigma^2\) is the observation
noise variance. The model parameters are therefore \(W\), \(\lambda\), and
\(\sigma^2\).

### Spike-and-slab mode activation

The key modeling step is a two-level spike-and-slab prior. For each mode \(j\)
and snapshot pair \(i\),

$$
p(\chi_{j,i}\mid \phi_{j,i}, z_{\chi,j,i})
=
(1-z_{\chi,j,i})\delta(\chi_{j,i})
+z_{\chi,j,i}\delta(\chi_{j,i}-\phi_{j,i}),
$$

$$
p(\psi_{j,i}\mid \phi_{j,i}, z_{\psi,j,i})
=
(1-z_{\psi,j,i})\delta(\psi_{j,i})
+z_{\psi,j,i}\delta(\psi_{j,i}-\phi_{j,i}),
$$

with

$$
\phi_{j,i}\sim \mathcal{CN}(0,1).
$$

The variable \(\phi_{j,i}\) is the latent value corresponding to a Koopman
eigenfunction. The binary variables
\(z_{\chi,j,i},z_{\psi,j,i}\in\{0,1\}\) control whether the \(j\)-th mode is
active for the current and next snapshot. If both are one, mode \(w_j\)
contributes throughout the interval. If both are zero, it does not contribute.
If one is one and the other is zero, the mode turns on or off within that
snapshot interval.

### Temporal prior on switches

The on/off variables are temporally structured with Gaussian-process-driven
Bernoulli probabilities:

$$
p(z_j\mid \gamma_j)
=
\prod_i \mathrm{Bernoulli}(z_{j,i};\Phi(\gamma_{j,i})),
$$

$$
\gamma_j \sim \mathcal{N}(\mu_j\mathbf{1},\Sigma_j),
$$

where \(\Phi\) is the standard normal CDF. The GP mean \(\mu_j\) controls the
overall tendency of mode \(j\) to be active or inactive, while the covariance
\(\Sigma_j\) encodes temporal smoothness through a user-chosen kernel. The paper
also discusses fast inference assumptions: with evenly spaced snapshots and a
translation-invariant kernel, the GP covariance becomes Toeplitz, which makes
inversion cheaper; multiple evenly spaced episodes yield block-Toeplitz
structure.

### Inference and learning

Learning uses an approximate EM procedure.

In the E-step, the posterior over

$$
\chi_{1:n}, \psi_{1:n}, \phi_{1:n}, z_{1:r}, \gamma_{1:r}
$$

is intractable, so the authors approximate it with expectation propagation
(EP). EP replaces each difficult factor in the posterior with an exponential
family site approximation and iteratively updates the site parameters by moment
matching.

In the M-step, point estimates of \(W\), \(\lambda\), and \(\sigma^2\) are
updated from posterior moments. The paper gives closed-form updates. In
compressed form, one key update is

$$
W \leftarrow E_1 E_2^{-1},
$$

where \(E_1\) and \(E_2\) are posterior expected cross-moment and second-moment
quantities involving \(x_i\) and \(\chi_i\). The eigenvalues \(\lambda\) are
then updated from moment matrices involving \(W\), \(\psi_i\), and \(y_i\), and
\(\sigma^2\) is updated by the expected reconstruction error for both \(x_i\)
and \(y_i\). The GP mean parameter \(\mu_j\) can also be updated by empirical
Bayes.

The authors explicitly note two practical issues: approximate EM can find only
local optima, and EP has no general convergence guarantee. They report that
initializing from standard DMD and damping EP updates helps in practice.

## Results and limitations

### Reported results

The paper reports two numerical examples.

First, it studies a superposition of two decaying traveling waves. One wave is
present globally in time, while the other appears only during a finite interval.
Standard DMD gives a distorted reconstruction because the sudden rise and fall
of the local wave violate the time-invariant model. FSDMD reconstructs the
components more accurately, captures the off interval of the local wave, and
recovers the eigenvalues of both waves.

Second, it analyzes simulated vorticity behind a cylinder. The trajectory begins
near an unstable equilibrium, passes through a transient regime, and approaches
a Karman vortex street limit cycle. FSDMD estimates mode activations that align
with this progression: rapidly decaying modes are active early, and
higher-frequency modes become active as the flow approaches the limit cycle.

These examples demonstrate modeling capability and interpretability of the
mode-switching structure. They are not broad statistical benchmarks.

### Limitations

- The number of dynamic modes \(r\) is effectively specified by the user in the
  experiments. The authors mention Bayesian nonparametrics as future work for
  more elegant automatic mode-count determination.
- GP kernel choices and hyperparameters are empirical. The method assumes that
  an appropriate temporal smoothness prior over mode activations can be chosen.
- Inference is relatively complex: approximate EM plus EP, with local-optimum
  risk and no EP convergence guarantee.
- The method operates in the DMD/probabilistic-DMD setting. It does not learn a
  nonlinear encoder/decoder or a sparse latent representation of the state.
- The switch variables are per dynamic mode, not basin-scale route variables.
  The method can describe transient mode activation, but it does not explicitly
  discover basin-support alignment or train route-local latent dynamics.
- The evidence is illustrative: two numerical examples, no large benchmark
  comparison, no comparison to sparse autoencoder representations, and no
  training/deployment study where basin counts and labels are withheld.

## Similarities to the current SKAE staged local \(K_c\) method

The closest conceptual overlap is that both FSDMD and the current SKAE staged
local \(K_c\) method reject the idea that a single homogeneous Koopman/DMD model
is always the right object for heterogeneous dynamics.

In FSDMD, heterogeneity is handled by allowing individual dynamic modes to be
active or inactive over time. In the current SKAE method, heterogeneity is
handled by first learning a global sparse Koopman autoencoder, then using the
dynamics of its learned supports to build route variables, and finally fitting
route-local affine maps.

More specifically, the shared themes are:

- **On/off structure as dynamical information.** FSDMD uses binary variables
  \(z_{\chi,j,i},z_{\psi,j,i}\) to decide whether mode \(j\) contributes at
  time \(i\). SKAE uses sparse latent supports, such as
  \(m_{\rm abs}(x)=\mathbf{1}\{|z_i|>10^{-3}\}\), as model-produced indicators
  of the active local dynamical regime.
- **Factorized switching rather than one monolithic model.** FSDMD factorizes
  the switch state across dynamic modes. SKAE factorizes representation through
  sparse active-coordinate identities, then compresses their support-flow fates
  into \(C_{\rm stab}\) route components.
- **Unsupervised route or activation discovery.** FSDMD does not require
  external labels for mode activations. SKAE does not use basin labels, basin
  counts, attractor identities, or trajectory-to-basin assignments to train the
  sparse KAE, construct \(C_{\rm stab}\), or train local affine maps.
- **Transient/local dynamics are central.** FSDMD is designed for temporally
  local events and transient regimes. SKAE is designed for multibasin systems
  where a single global latent matrix may mix incompatible local dynamics, and
  where support changes can indicate basin-scale or attractor-scale structure.
- **Both are Koopman-motivated hybridizations.** FSDMD hybridizes DMD with a
  probabilistic switching model. SKAE hybridizes a Koopman autoencoder with
  sparse support objects and downstream local affine latent predictors.
- **Both create interpretable discrete objects.** FSDMD's object is an on/off
  time series for each dynamic mode. SKAE's object is a support family or stable
  support component, especially \(C_{\rm stab}\), which is read as a
  support-flow fate rather than a ground-truth basin label.

These similarities make Takeishi et al. important related work. It is relevant
whenever our paper discusses switching Koopman/DMD methods, on/off mode
activation, or unsupervised temporal decomposition of heterogeneous dynamics.

## Differences from the current SKAE staged local \(K_c\) method

Despite the shared high-level motivation, the technical object and claim are
substantially different.

### Representation

FSDMD is a probabilistic DMD model over observed snapshot pairs. It estimates
dynamic modes \(W\), eigenvalues \(\lambda\), latent coefficients, and mode
activation variables. It does not learn an encoder/decoder pair.

The current SKAE method first trains one global sparse Koopman autoencoder:

$$
z_t = \mathrm{Enc}(x_t), \qquad
\hat x_t = \mathrm{Dec}(z_t), \qquad
z_{t+1}\approx K z_t.
$$

The encoder, decoder, and global \(K\) are shared across all trajectories during
stage 1. Sparse supports are not prescribed as switch variables in a generative
model; they emerge from the learned representation under the sparse KAE
objective.

### Switching or routing object

FSDMD's switch variable is per mode and per time. The active set says which
dynamic modes contribute to the observation at a given time.

SKAE's route is derived after global training from the empirical dynamics of
learned sparse supports. The target routing object is

$$
C_{\rm stab},
$$

a stable support component built from support-flow fates. The construction is:

1. Encode training trajectories with the frozen stage-1 encoder.
2. Convert each latent code to a sparse support mask.
3. Group exact masks into fine base support-family labels \(u_t\).
4. Estimate empirical support transitions \(u_t\to u_{t+1}\).
5. Identify recurrent support-flow components and assign observed support
   states by their absorption fate.
6. Use \(c=C_{\rm stab}(u_t)\) as the route.

This route is not a direct mode activation vector, not a basin label, and not a
fixed-point estimate. It is a dynamical quotient of the learned support process.

### Local dynamics

FSDMD keeps a single modal dictionary \(W\) and a single diagonal eigenvalue
matrix \(\Lambda\). Time variation is expressed by switching contributions of
the modes on and off. It does not fit one full transition matrix per discovered
route.

The SKAE staged method freezes the learned representation and then trains local
affine latent maps:

$$
T_c(z) = d_c + K_c(z-\bar z_c),
$$

where \(c=C_{\rm stab}(u_t)\), \(\bar z_c\) is a source-center statistic for the
route, \(K_c\) is a route-specific linear map, and \(d_c\) is a learned target
center. This is closer to a learned bank of local affine latent predictors,
selected by support-flow fate, than to switching on and off columns of a single
DMD mode matrix.

### Training structure

FSDMD learns modes, eigenvalues, latent coefficients, switch variables, and GP
switch priors jointly through approximate EM/EP.

The current SKAE method is staged:

1. Train one global sparse Koopman autoencoder with shared
   \(\mathrm{Enc}\), \(\mathrm{Dec}\), and \(K\).
2. Derive \(C_{\rm stab}\) from learned sparse support-flow fates, without
   basin labels or known basin counts.
3. Freeze the representation.
4. Train local affine maps \(T_c(z)=d_c+K_c(z-\bar z_c)\) while preserving the
   shared encoder/decoder and using the frozen global \(K\) as fallback or
   initialization.

This staging matters. The route is not optimized as a latent switch variable
during the first-stage model fit; it is extracted from the learned sparse
representation after the global model has already been trained.

### Assumptions about counts and labels

FSDMD is unsupervised with respect to switch activations, but the number of
dynamic modes \(r\) is chosen in the experiments. It models time variation by
assuming that a finite set of reusable dynamic modes exists and that their
activation probabilities have a GP-temporal structure.

The SKAE staged local \(K_c\) method is designed around a different
training/deployment assumption: the learner does not know the number of basins,
does not know which trajectories belong to which basin, and is not given basin
labels or basin counts. Known basin labels and counts are acceptable only for
benchmark evaluation. The route count emerges from the learned support-flow
construction, not from an input regime count.

### Target phenomenon

FSDMD targets time-variant systems in which a temporal subset of dynamic modes
explains the data. Its examples emphasize local waves and transient fluid-flow
modes.

SKAE targets multibasin Koopman learning and basin-support alignment. The
question is not only whether a transient mode turns on, but whether a sparse
Koopman representation can expose basin-scale support identities and use those
identities to route local latent dynamics.

### Supervision and interpretability

FSDMD's interpretability is mode-centered: the output includes eigenvalues,
dynamic modes, and time-varying on/off activation traces.

SKAE's interpretability is support-centered: active latent coordinate identities
are tested as regime variables. The paper's intended support object is not a
post-hoc label attached to trajectories, but an inspectable structure produced
by the learned sparse representation itself. The comparison target is therefore
not just "a switching Koopman model"; it is a method that turns learned sparse
support dynamics into a basin-scale routing interface.

## Novelty implications for our paper

Takeishi et al. should be cited as directly relevant prior work for unsupervised
switching DMD/Koopman analysis of time-varying systems. It weakens any overly
broad novelty statement such as:

- "We are the first to use switching in Koopman/DMD models."
- "We are the first to use on/off sparse variables for dynamic modes."
- "We are the first to model time-varying Koopman structure without labels."

Those claims would be inaccurate or at least too broad. FSDMD already uses
factorial binary activations of dynamic modes, GP-temporal structure over those
activations, and unsupervised approximate inference.

The defensible novelty of the current SKAE paper is different and should be
phrased more narrowly:

- We train a sparse Koopman autoencoder representation, not a probabilistic DMD
  model over fixed observables.
- We treat sparse latent supports as inspectable, model-produced regime
  variables and evaluate their basin-support alignment.
- We do not assume basin labels, basin counts, or trajectory-to-basin
  assignments in the training/deployment setting.
- We derive \(C_{\rm stab}\) routes from learned sparse support-flow fates after
  global training, rather than introducing an explicit switch process in the
  model likelihood.
- We freeze the shared representation and train route-local affine latent maps
  \(T_c(z)=d_c+K_c(z-\bar z_c)\), showing that the discovered support-stable
  routes can select useful local predictors.

The clean positioning is:

> FSDMD shows that time-varying Koopman/DMD structure can be modeled by
> unsupervised factorial on/off activation of dynamic modes. Our method instead
> asks whether a learned sparse Koopman autoencoder can expose basin-scale
> support structure without basin metadata, and whether the support-flow fates
> of that representation can route a downstream bank of local affine latent
> dynamics.

This preserves the contribution while acknowledging the closest prior idea:
binary/factorial switching in Koopman-motivated DMD is not new, but deriving
basin-scale local-model routes from learned sparse KAE support-flow fates is a
distinct methodological claim.
