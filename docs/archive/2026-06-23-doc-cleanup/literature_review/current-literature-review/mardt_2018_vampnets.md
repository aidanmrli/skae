# Mardt et al. (2018): VAMPnets for Deep Learning of Molecular Kinetics

## Full Citation

Mardt, A., Pasquali, L., Wu, H., & Noe, F. (2018). VAMPnets for deep learning of molecular kinetics. *Nature Communications*, 9, Article 5. https://doi.org/10.1038/s41467-017-02388-1

## Sources Read

- Official article page: https://www.nature.com/articles/s41467-017-02388-1
- Official PDF: https://www.nature.com/articles/s41467-017-02388-1.pdf
- Official supplementary information: https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-017-02388-1/MediaObjects/41467_2017_2388_MOESM1_ESM.pdf
- Official author correction: https://www.nature.com/articles/s41467-018-06999-0

The author correction only adds an omitted funding acknowledgement and does not change the technical claims analyzed here.

## Problem and Motivation

Mardt et al. address the problem of learning long-timescale molecular kinetics from molecular dynamics trajectories. The standard workflow at the time was a manually engineered Markov state model pipeline: choose molecular features, reduce dimension, discretize or cluster the reduced representation, estimate a transition matrix or related operator, and then coarse-grain to interpretable metastable states. The paper argues that each step requires domain expertise and can introduce modeling error; a poor early feature choice can make later kinetic modeling fail.

The authors propose to replace this pipeline with an end-to-end neural architecture trained directly on time-lagged molecular configurations. The aim is to learn a transformation from high-dimensional molecular coordinates into a small number of kinetic features or fuzzy metastable-state memberships such that the transformed dynamics are well approximated by a finite-dimensional Koopman model or Markov state model.

The motivating object is therefore a learned kinetic model of molecular metastability. The paper is not about sparse latent supports, basin-support alignment, reconstruction, or route-local dynamics. Its central claim is that a deep network can learn the molecular featurization, dimension reduction, fuzzy clustering, and coarse kinetic model together by optimizing a variational score for Markov processes.

## Method Details

### Time-Lagged Feature Learning

The method starts from time-lagged trajectory pairs \((x_t, x_{t+\tau})\). A VAMPnet has two neural-network lobes that map these configurations into feature vectors

$$
\chi_0(x_t), \qquad \chi_1(x_{t+\tau}).
$$

The intended finite-dimensional Koopman relation is

$$
\mathbb{E}\left[\chi_1(x_{t+\tau})\right]
\approx
K^\top \mathbb{E}\left[\chi_0(x_t)\right].
$$

For a fixed pair of feature maps, the empirical covariance matrices are

$$
C_{00} = \mathbb{E}_t[\chi_0(x_t)\chi_0(x_t)^\top],
$$

$$
C_{01} = \mathbb{E}_t[\chi_0(x_t)\chi_1(x_{t+\tau})^\top],
$$

$$
C_{11} = \mathbb{E}_{t+\tau}[\chi_1(x_{t+\tau})\chi_1(x_{t+\tau})^\top].
$$

The least-squares estimate of the Koopman matrix is

$$
K = C_{00}^{-1} C_{01}.
$$

The authors emphasize that minimizing one-step least-squares prediction error is not enough, because a constant feature can minimize prediction error while discarding all useful dynamics. They therefore optimize the feature maps using the VAMP variational principle.

### VAMP-2 Objective

The training objective is the VAMP-2 score

$$
\hat R_2[\chi_0,\chi_1]
=
\left\|
C_{00}^{-1/2} C_{01} C_{11}^{-1/2}
\right\|_F^2.
$$

This score is maximized when the learned feature subspaces approximate the dominant left and right singular-function subspaces of the Koopman operator. In the reversible equilibrium setting, the formulation specializes to the variational approach for conformation dynamics; in the more general setting it can handle nonreversible and nonstationary Markov processes.

In their applications, the two lobes are usually tied so that \(\chi_0=\chi_1=\chi\). The output layer is commonly a Softmax layer:

$$
\chi_i(x) \ge 0, \qquad \sum_i \chi_i(x)=1.
$$

With Softmax outputs, the coordinates are interpretable as fuzzy memberships in metastable states. The corresponding \(K(\tau)\) estimated from the transformed data is then interpreted as a transition matrix for a fuzzy Markov state model.

### Model Validation

The learned model is evaluated using standard kinetic validation tools. If

$$
K(\tau) r_i = r_i \lambda_i(\tau),
$$

then implied timescales are

$$
t_i(\tau)
=
-\frac{\tau}{\log |\lambda_i(\tau)|}.
$$

A plausible lag time is one where the implied timescales are approximately stable as \(\tau\) varies. The authors also use the Chapman-Kolmogorov consistency check

$$
K(n\tau) \approx K(\tau)^n,
$$

which tests whether the fitted model's long-time predictions agree with models estimated at longer lag times.

## Results and Limitations

### Main Results

On an asymmetric one-dimensional double well, VAMPnets learn fuzzy states that concentrate resolution near the transition region, approximate the dominant Koopman eigenfunction, and pass implied-timescale and Chapman-Kolmogorov checks.

On a five-dimensional toy folding model, the network receives five Cartesian coordinates but the relevant slow coordinate is the nonlinear radius \(r=|x|\). A two-output VAMPnet recovers the folded/unfolded distinction and effectively learns this nonlinear reaction coordinate from the VAMP-2 objective.

On alanine dipeptide, the network receives aligned Cartesian coordinates of the ten heavy atoms, not the known backbone torsions. A six-output VAMPnet learns metastable sets corresponding to free-energy minima in the \((\phi,\psi)\) Ramachandran space. A bottleneck version with two internal nodes yields activations strongly correlated with \(\phi\) and \(\psi\), with reported Pearson correlations of 0.95 and 0.92. The comparison to a standard MSM pipeline shows stronger VAMP-2 scores for VAMPnets when fewer than 20 states are used; the standard MSM can improve with more states but then needs additional coarse-graining for interpretability.

On NTL9 folding, a five-output VAMPnet trained on contact-map features gives interpretable folded, unfolded, intermediate, and misfolded states. The relaxation timescales are comparable to a 40-state TICA plus k-means MSM, while the VAMPnet representation is much smaller and easier to interpret.

### Limitations

The method still requires choosing the number of output states, and that choice has a direct effect on the learned discretization. A Softmax output with \(k\) states has only \(k-1\) degrees of freedom because of normalization, so resolving more relaxation processes requires more output states.

Training can be fragile. For alanine dipeptide with six output states, the authors report that even in the favorable lag-time range the success rate remains below 40%, mainly because many runs fail to recover a rare slower process.

The objective is kinetic and variational, not reconstructive. VAMPnets learn features useful for approximating dominant Koopman singular functions and fuzzy metastable states, but they do not require an encoder-decoder representation that preserves information for reconstruction.

The learned states are membership coordinates, not sparse supports. The paper does not attempt to align basins with sparse latent supports, does not derive routes from support-flow fates, and does not train route-local affine dynamics after freezing a shared representation.

The authors also note that VAMPnets lack several mature extensions available in MSM practice, including multi-ensemble modeling, integration of experimental observables through augmented Markov models, and likelihood-based statistical error estimators.

## Similarities to the Current SKAE Staged Local \(K_c\) Method

Both VAMPnets and the current SKAE staged local \(K_c\) method are neural approaches to discovering useful coordinates for nonlinear dynamical systems from trajectory data. In both cases, the high-dimensional observed state is transformed into a lower-dimensional representation where some form of approximately linear time evolution becomes meaningful.

Both methods are Koopman-motivated. VAMPnets explicitly optimize a variational score whose optimum corresponds to dominant Koopman singular-function subspaces. SKAE trains a sparse Koopman autoencoder with a shared latent linear operator \(K\), so the first-stage model also asks for a representation in which latent evolution is approximately linear.

Both methods use time-lagged data without requiring explicit class labels for the training pairs. VAMPnets train on \((x_t,x_{t+\tau})\) pairs through the VAMP score. SKAE trains a global sparse Koopman autoencoder on trajectories and then uses learned sparse support-flow fates to derive \(C_{\text{stab}}\) routes, without assuming basin labels or a known number of basins during training-time method design.

Both methods pursue interpretability by compressing the dynamics into a small number of meaningful objects. VAMPnets interpret Softmax outputs as fuzzy metastable-state memberships and interpret \(K(\tau)\) as a fuzzy transition matrix. SKAE interprets sparse latent supports and support-flow fates as basin-aligned dynamical structure, then assigns route-local dynamics through \(K_c\).

Both methods can be framed as alternatives to a hand-engineered pipeline. VAMPnets replace featurization, dimension reduction, clustering, and MSM estimation with one neural training objective. SKAE replaces direct dependence on known basin labels or fixed basin counts with a staged procedure: first learn a global sparse representation and global Koopman dynamics, then infer stable routes from the learned representation, then refine dynamics locally.

## Differences from the Current SKAE Staged Local \(K_c\) Method

The main learned object is different. VAMPnets learn fuzzy kinetic state memberships optimized for the VAMP-2 score. SKAE learns a reconstructive sparse latent state \(z\), where the target interpretability structure is basin-support alignment: each basin should map to a distinct sparse support in latent space.

The routing mechanism is different. VAMPnet routing is effectively the Softmax membership vector learned directly by the network, with the output dimension specified by the user. SKAE derives \(C_{\text{stab}}\) routes after global training from learned sparse support-flow fates. This distinction matters because SKAE's routing is not a predeclared Softmax partition over a fixed number of kinetic states; it is inferred from the trajectory of supports in the learned latent representation.

The assumptions about basin counts differ. VAMPnets require the user to choose the number of output nodes, and that choice controls the resolution of the metastable decomposition. In the SKAE training/deployment setting, the intended method does not assume that the number of basins is known in advance and does not assume access to basin labels. Known basin counts and labels can still be used for benchmark evaluation, but they are not part of the training-time method design.

The treatment of dynamics differs. VAMPnets estimate one global Koopman or fuzzy MSM transition matrix \(K(\tau)\) in the learned feature space. SKAE first trains one global sparse Koopman autoencoder with shared encoder, shared decoder, and global \(K\), but then freezes the representation and trains local affine latent maps

$$
T_c(z) = d_c + K_c(z-\bar z_c).
$$

This local \(K_c\) stage is not present in VAMPnets. VAMPnets may output multiple states, but the transition model is still a single global matrix over those fuzzy states. SKAE explicitly models route- or basin-conditioned local affine dynamics after the representation has been learned.

The training decomposition differs. VAMPnets jointly learn the representation and kinetic model through a single variational objective. SKAE is staged: global sparse autoencoder training first, support-flow route discovery second, representation freezing third, and local affine \(K_c\) fitting fourth. This staged design is important because it separates representation discovery from local dynamics estimation.

The role of reconstruction differs. VAMPnets do not decode learned features back to the original configuration; reconstruction is not part of the objective. SKAE is an autoencoder, so the latent representation must both support Koopman prediction and retain enough information to reconstruct the observed state.

The interpretability unit differs. VAMPnets produce metastable memberships and transition probabilities between learned fuzzy states. SKAE aims to produce sparse support identities, support transitions, stable routes, and route-local latent dynamics. These are related in spirit but not the same type of explanation.

The supervision signal differs in a subtle but important way. VAMPnets are self-supervised by a variational score on lagged pairs, but the output-state count is a manually chosen structural prior. SKAE is also self-supervised in the first stage, but its downstream \(C_{\text{stab}}\) routes are derived from learned support-flow fates rather than from a Softmax output layer whose dimensionality specifies the kinetic partition.

## Novelty Implications for Our Paper

VAMPnets are strong prior work for the broad idea that neural networks can learn Koopman-compatible kinetic representations from time-lagged trajectory data. Our paper should not claim novelty at the level of "deep learning for Koopman molecular kinetics," "end-to-end learned kinetic states," or "neural replacement for an MSM feature-engineering pipeline." Those claims are already central to Mardt et al.

VAMPnets also reduce novelty for the narrower claim that learned neural features can expose metastable structure without explicit state labels. Their alanine dipeptide and NTL9 results show that VAMP-optimized networks can discover physically meaningful kinetic decompositions from molecular coordinates or contact features.

The SKAE staged local \(K_c\) method remains meaningfully distinct if the paper frames the contribution around the following points:

1. Basin-support alignment rather than Softmax metastable-state membership. The intended representation-level claim is that basins map to unique sparse supports in \(z\), not merely that a network assigns configurations to fuzzy kinetic states.

2. Route discovery from learned sparse support-flow fates. The \(C_{\text{stab}}\) routes are inferred after global training from support dynamics, without training-time basin labels or fixed basin counts.

3. A staged global-to-local dynamics procedure. The method first learns one shared encoder, decoder, and global \(K\), then freezes the representation and trains local affine maps \(T_c(z)=d_c+K_c(z-\bar z_c)\). VAMPnets do not have an analogous local \(K_c\) refinement stage.

4. Reconstruction-constrained Koopman representation learning. SKAE's latent coordinates must support both reconstruction and dynamical prediction, whereas VAMPnets optimize kinetic feature quality directly.

5. Local affine latent dynamics as a paper-facing mechanism. If the results show that the global sparse representation discovers basin-aligned supports and that route-local \(K_c\) maps improve or clarify dynamics, this is not preempted by VAMPnets.

The safest positioning is to cite VAMPnets as foundational neural Koopman/MSM work and then state that SKAE addresses a different representational and modeling problem: discovering sparse basin-aligned latent supports and using their support-flow fates to define unsupervised routes for local affine Koopman dynamics. The novelty should be argued at the level of sparse support geometry, unsupervised route construction without known basin counts, and staged local affine refinement, not at the level of neural Koopman learning in general.
