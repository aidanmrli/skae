# Becker-Ehmck et al. (2019): Switching Linear Dynamics for Variational Bayes Filtering

## Full Citation

Becker-Ehmck, P., Peters, J., & van der Smagt, P. (2019). *Switching Linear Dynamics for Variational Bayes Filtering*. Proceedings of the 36th International Conference on Machine Learning, Proceedings of Machine Learning Research, 97, 553--562.

## Sources Read

Primary sources:

- PMLR paper page: <https://proceedings.mlr.press/v97/becker-ehmck19a.html>
- PMLR main PDF: <https://proceedings.mlr.press/v97/becker-ehmck19a/becker-ehmck19a.pdf>
- PMLR supplementary PDF: <https://proceedings.mlr.press/v97/becker-ehmck19a/becker-ehmck19a-supp.pdf>
- arXiv metadata page: <https://arxiv.org/abs/1905.12434>

This note uses the ICML/PMLR paper and supplement as the article sources. The arXiv page was used only to cross-check bibliographic metadata.

## Problem and Motivation

The paper studies unsupervised system identification for nonlinear dynamical systems from partial or high-dimensional observations. The motivating applications are model predictive control and model-based reinforcement learning, where a learned dynamics model must both predict accurately and expose useful latent structure.

The central premise is the switching linear dynamical systems (SLDS) view: many nonlinear systems can be approximated by splitting trajectories into subsequences governed by simpler local linear dynamics. In this framing, a discrete or relaxed switching variable selects the local transition law. The authors argue that such switching variables can encode meaningful physical structure, such as wall contacts in a maze, joint constraints in a robot arm, or collisions in image-based bouncing-ball data.

This is not a Koopman paper. It does not discuss finite-dimensional Koopman invariant subspaces, sparse latent supports, basin-support alignment, or attractor-basin discovery. Its relevant connection to our work is broader: it is a learned latent-dynamics model in which local linear laws are selected by unsupervised latent regime variables.

## Method Details

### Classical SLDS Starting Point

The paper begins from a standard SLDS model with continuous latent state \(z_t\), observed state \(x_t\), optional control \(u_t\), and switch variable \(s_t\). A classical switch-conditioned linear transition has the form

\[
z_t = A(s_t) z_{t-1} + B(s_t) u_{t-1} + \epsilon(s_t),
\]

\[
x_t = H(s_t) z_t + \eta(s_t).
\]

Here \(s_t\) chooses one of \(M\) local linear systems. The paper replaces parts of this classical model with neural variational components while preserving the idea that local transition dynamics are linear after conditioning on the switch.

### Generative Model

The learned latent transition is Gaussian and switch-conditioned:

\[
p_\theta(z_t \mid z_{t-1}, s_t, u_{t-1})
= \mathcal N(\mu_t, \Sigma_t),
\]

with

\[
\mu_t = A_\theta(s_t) z_{t-1} + B_\theta(s_t) u_{t-1},
\qquad
\Sigma_t = Q_\theta(s_t).
\]

The observation model is nonlinear:

\[
p_\theta(x_t \mid z_t),
\]

implemented as a neural decoder. A key design choice is that \(p_\theta(x_t \mid z_t)\) is not directly conditioned on \(s_t\). This encourages \(s_t\) to explain dynamics rather than merely helping reconstruct observations.

The switch prior is also learned:

\[
p_\theta(s_t \mid s_{t-1}, z_{t-1}, u_{t-1}).
\]

For discrete switching, the paper uses the Concrete / Gumbel-softmax relaxation. If the relaxed switch is \(s_t \in \Delta^{M-1}\), then the local transition matrices are combined softly:

\[
A_\theta(s_t) = \sum_{i=1}^M s_t^{(i)} A_i,
\]

\[
B_\theta(s_t) = \sum_{i=1}^M s_t^{(i)} B_i,
\qquad
Q_\theta(s_t) = \sum_{i=1}^M s_t^{(i)} Q_i.
\]

This means the forward model is often a convex mixture of base linear systems, not a hard selected expert.

### Inference Model

The inference model combines a transition prediction with an inverse-measurement prediction. For the continuous latent state, the variational posterior is built as a product of two Gaussian factors:

\[
q_\phi(z_t \mid z_{t-1}, s_t, x_{\ge t}, u_{\ge t-1})
\propto
q_{\rm meas}(z_t \mid x_{\ge t}, u_{\ge t})
q_{\rm trans}(z_t \mid z_{t-1}, s_t, u_{t-1}).
\]

Because both terms are Gaussian, the product gives a Gaussian posterior with precision-weighted mean and variance. This structured inference design shares transition parameters with the generative model so that reconstruction gradients pass through the transition model.

For switch inference, the paper similarly combines a transition prediction and an inverse-measurement prediction. In the Concrete case,

\[
q_\phi(s_t \mid s_{t-1}, z_{t-1}, x_{\ge t}, u_{\ge t-1})
= {\rm Concrete}(\alpha_t, \lambda_{\rm posterior}),
\]

where

\[
\alpha_t = \gamma_t \alpha_{\rm trans}
          + (1-\gamma_t)\alpha_{\rm meas}.
\]

The learned gate \(\gamma_t\) controls how much the posterior switch assignment trusts the dynamics prior versus the measurement evidence. The measurement-side switch encoder is a backward recurrent model in the offline setting, or an MLP in the online filtering setting.

### Objective and Training

Training maximizes an evidence lower bound:

\[
\mathcal L_{\theta,\phi}
=
\mathbb E_{q_\phi}
\left[
\log p_\theta(x_{1:T} \mid z_{1:T}, s_{2:T}, u_{1:T})
\right]
-
{\rm KL}
\left(
q_\phi(z_{1:T},s_{2:T}\mid x_{1:T},u_{1:T})
\;\|\;
p_\theta(z_{1:T},s_{2:T}\mid u_{1:T})
\right).
\]

The per-step form decomposes into reconstruction likelihood, a switch KL term, and a latent-transition KL term. The authors downweight the switch KL with a factor \(\beta < 1\), use reparameterization for continuous and Concrete variables, and approximate the KL between Concrete variables by Monte Carlo samples.

The supplement makes clear that Concrete-switch training is delicate. It requires temperature schedules, KL scaling, and empirical tuning. The authors also study a normally distributed switch variable variant, where the switch latent is mapped through a linear layer and softmax to produce transition mixing coefficients. In the main quantitative table, this Normal variant is often the strongest model.

## Results and Limitations

### Results

The paper evaluates on simulated physical systems with partial or image observations:

- Multi-agent maze: the learned continuous latent state recovers hidden velocities from position-only observations, and switch variables encode wall and free-space structure. The reported linear velocity recovery has \(R^2 \approx 0.92\). A wall-interaction classifier trained from switch latents obtains F1 around \(0.46\).
- Roboschool Reacher: switch variables capture joint-collision structure. A classifier based on the switch latent obtains F1 around \(0.53\), and the model improves prediction error over recurrent and DVBF-style baselines in their table.
- Image ball-in-box: the model performs well on longer-horizon prediction from \(32 \times 32\) binary image observations and is reported to outperform KVAE on longer trajectories, though KVAE is stronger at short horizons.
- FitzHugh-Nagumo: the model matches the predictive performance of tree-structured SLDS on the normalized multi-step prediction comparison.
- Discretization-scale study: Concrete/discrete switch variables become less favorable as the temporal discretization interval grows. The Normal switch variant is more flexible because it can scale transition mixtures more freely.

### Limitations

The model assumes a chosen switch dimension or number of base systems. It does not infer the number of regimes in the deployment sense relevant to our basin-support setting.

The regime variable is trained as an explicit latent switch in an end-to-end probabilistic model. This is not the same as discovering regimes from the active support of a learned sparse coordinate representation.

The forward transition is often a soft mixture of matrices rather than a hard-selected local law. This improves differentiability but weakens the interpretation that a single local linear system is active.

The paper's interpretable regimes are local physical interaction events: walls, free space, collisions, or joint constraints. They are not basins of attraction, recurrent support-flow fates, or attractor-relative dynamical equivalence classes.

The authors themselves report optimization sensitivity for Concrete switches. The supplement notes the need for KL downweighting and Monte Carlo KL approximation, and the main results show that the Normal switch variant can outperform the Concrete variant.

The paper does not evaluate support stability, basin purity, route coverage, component count recovery, or downstream local-law fitting from a frozen representation.

## Similarities to the Current SKAE Staged Local \(K_c\) Method

The broad conceptual similarity is that both approaches use latent-variable structure to organize nonlinear dynamics into simpler local linear pieces.

Both methods learn a latent state \(z_t\) from observations and use linear or affine latent transitions to model nonlinear dynamics. Becker-Ehmck et al. use switch-conditioned linear Gaussian latent transitions. The SKAE staged method first learns a sparse Koopman latent representation with one shared global \(K\), then trains route-conditioned local affine maps

\[
T_c(z) = d_c + K_c(z-\bar z_c).
\]

Both methods use unsupervised regime information. Becker-Ehmck et al. infer switch variables \(s_t\) without labels for walls, collisions, or contacts. SKAE derives \(C_{\rm stab}\) routes from learned sparse support-flow fates without basin labels, basin counts, or trajectory-to-basin assignments.

Both methods view local linear structure as a way to improve prediction and interpretability. In Becker-Ehmck et al., the switch variables are expected to reveal constraints and interaction modes. In SKAE, the support-flow components are expected to reveal basin-support alignment and select basin-appropriate local affine laws.

Both methods can be read as learned alternatives to hand-specified hybrid system partitions. Neither requires an externally supplied regime label for each training trajectory.

Both methods have a route-dependent transition bank. Becker-Ehmck et al. use matrices \(\{A_i,B_i,Q_i\}_{i=1}^M\) mixed according to \(s_t\). SKAE trains a bank of local \(K_c\) maps after route construction, with a fallback to the frozen global \(K\) when no reliable local route is available.

Both approaches are relevant prior art against any claim that "unsupervised latent switching plus local linear dynamics" is new. That high-level idea is clearly established before SKAE.

## Differences from the Current SKAE Staged Local \(K_c\) Method

The most important difference is the route variable.

In Becker-Ehmck et al., routing is performed by an explicit latent switch \(s_t\) with a learned prior and posterior. The switch is part of the generative model and is optimized directly through the ELBO. In SKAE, the route is not an added switch latent. It is derived after representation learning from the active coordinate support of the learned sparse latent \(z\). The \(C_{\rm stab}\) route is built from the empirical flow of support states into recurrent support components.

The second major difference is the training sequence.

Becker-Ehmck et al. train representation, switch inference, decoder, and transition dynamics jointly. SKAE deliberately separates the stages:

1. Train one global sparse Koopman autoencoder with shared encoder, decoder, and global \(K\).
2. Derive \(C_{\rm stab}\) routes from learned sparse support-flow fates without basin labels or known basin counts.
3. Freeze the encoder, decoder, and global \(K\).
4. Train local affine latent maps \(T_c(z)=d_c+K_c(z-\bar z_c)\) using decoded rollout MSE.

This staging matters for interpretation. SKAE asks whether the sparse representation learned by a global Koopman objective already contains basin-fate information that can be harvested post hoc. Becker-Ehmck et al. do not test whether a single global linear latent model spontaneously produces support-defined regime structure.

The third difference is the notion of regime.

Becker-Ehmck et al. target local interaction regimes such as wall contacts, free-space movement, and joint collisions. These are local physical conditions. SKAE targets basin-support alignment: the desired object is that each basin of attraction maps to a unique sparse support fate. \(C_{\rm stab}\) is a dynamical equivalence object over support trajectories, not a classifier for instantaneous contacts.

The fourth difference is regime count.

Becker-Ehmck et al. require a chosen switch dimension or number of base systems. Even when the switch is relaxed, the model architecture fixes the available switch space. In SKAE, the number of stable support-flow routes is not fixed to the benchmark basin count. It emerges from the learned support-transition graph, with basin metadata used only afterward for evaluation on benchmark systems.

The fifth difference is supervision and deployment assumptions.

Both methods are unsupervised with respect to physical regime labels, but SKAE has a sharper constraint: in the intended training/deployment setting, the method does not know the number of basins or which trajectories belong to which basin. The \(C_{\rm stab}\) construction preserves that constraint. Becker-Ehmck et al. are not designed around unknown basin counts or attractor-basin discovery.

The sixth difference is the role of sparsity.

Sparsity is central to SKAE. The route is literally a function of which latent coordinates are active and how those supports flow over time. Becker-Ehmck et al. do not use sparse coordinate support as the regime object. Their switch variable is a separate latent variable with its own inference machinery.

The seventh difference is relation to Koopman modeling.

SKAE is framed around sparse Koopman autoencoders and the difficulty of representing multibasin nonlinear systems with one finite-dimensional global linear operator. The staged local maps are downstream local Koopman-style charts. Becker-Ehmck et al. are framed as variational SLDS / deep state-space modeling. Their local matrices are not presented as Koopman operators, and their paper does not study Koopman-theoretic basin limitations.

The eighth difference is route differentiability.

Becker-Ehmck et al. make routing differentiable through Concrete or continuous relaxations. SKAE's current staged \(C_{\rm stab}\) route is post-hoc and non-differentiable: the latent is converted to a support mask, assigned to a support-flow component, and used to select a local map. That makes the SKAE staged result a cleaner diagnostic of what the frozen representation already exposes, but it also means the current route construction is not an end-to-end variational switch model.

## Novelty Implications for Our Paper

Becker-Ehmck et al. should be cited as relevant prior work on learned switching linear latent dynamics, especially because it combines variational inference, latent state-space modeling, and unsupervised switch variables for local linear transitions.

The paper narrows what we can safely claim. We should not claim that SKAE is the first method to learn unsupervised latent regimes, the first to combine deep latent-variable models with switching linear dynamics, or the first to use local linear latent transitions for nonlinear system identification.

The paper does not undermine the central SKAE novelty if our contribution is stated precisely. The defensible novelty is not "switching linear dynamics." It is:

- learning a sparse Koopman autoencoder with one shared global latent transition first;
- treating learned sparse supports as the regime-bearing object;
- constructing \(C_{\rm stab}\) routes from support-flow fates without basin labels, trajectory-to-basin assignments, or known basin counts;
- using benchmark basin labels only for evaluation of basin-support alignment;
- freezing the learned representation and testing whether support-flow routes select useful local affine latent maps;
- targeting basin-support alignment rather than contact-mode or collision-mode discovery.

For coauthor positioning, the clean statement is:

SKAE is related to variational SLDS models such as Becker-Ehmck et al. because both use latent local linear dynamics and unsupervised regime variables. However, SKAE's regime variable is not an explicit learned switch. It is a post-hoc dynamical object induced by sparse Koopman supports. The paper's main claim should therefore be about sparse-support discovery of basin-fate structure and its use as a label-free local-law route, not about introducing switching linear latent dynamics as a general modeling idea.

This distinction also affects how the staged local \(K_c\) result should be presented. The staged experiment is strongest as evidence that the sparse Koopman representation contains reusable basin-fate routing information. It is weaker as a broad claim that local linear transition banks are new or universally superior, because SLDS and variational switching models already establish that family of ideas.
