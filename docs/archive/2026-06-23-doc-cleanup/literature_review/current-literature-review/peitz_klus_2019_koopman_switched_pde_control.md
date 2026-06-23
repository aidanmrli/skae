# Peitz and Klus (2019): Koopman Model Reduction for Switched-System PDE Control

## Full citation

Sebastian Peitz and Stefan Klus. "Koopman operator-based model reduction for switched-system control of PDEs." *Automatica* 106, 184-191, 2019. DOI: [10.1016/j.automatica.2019.05.016](https://doi.org/10.1016/j.automatica.2019.05.016).

## Sources read

- Published article metadata and DOI: [Paderborn University Research Information System](https://ris.uni-paderborn.de/record/10593).
- Published article metadata and abstract: [Heriot-Watt Research Portal](https://researchportal.hw.ac.uk/en/publications/koopman-operator-based-model-reduction-for-switched-system-contro/).
- Full text read from the authors' arXiv version: [arXiv:1710.06759v2](https://arxiv.org/abs/1710.06759), PDF at [https://arxiv.org/pdf/1710.06759](https://arxiv.org/pdf/1710.06759).

The analysis below is based on the author-posted arXiv full text, cross-checked against the published Automatica bibliographic record.

## Problem and motivation

Peitz and Klus address fast optimal and feedback control for nonlinear systems governed by PDEs. Their motivating setting is PDE-constrained control, where the state \(y(\cdot,t)\) may be infinite-dimensional before discretization and where direct model predictive control (MPC) is often too expensive because each online optimization step requires many high-fidelity PDE solves.

The baseline control problem is

\[
\min_{u \in \mathcal U} J(y)
= \min_{u \in \mathcal U} \int_{t_0}^{t_e} L(y(\cdot,t))\,dt
\]

subject to

\[
\dot y(\cdot,t)=G(y(\cdot,t),u(t)), \qquad y(\cdot,0)=y^0.
\]

Classical projection-based reduced-order modeling, especially POD/Galerkin methods, is a natural alternative to full PDE solves, but the paper emphasizes that such models may require many modes or extra stabilization/calibration for complex nonlinear dynamics. The authors propose Koopman operator-based reduced-order models (K-ROMs) as an alternative because Koopman models can give linear predictors for selected observables rather than requiring reconstruction of the full PDE state.

The central control insight is that the controller may only need a low-dimensional observation of the system, not the full state. For example, in the PDE examples, the controlled objective is expressed through sensor measurements, lift/drag coefficients, or other low-dimensional quantities. This allows the online control problem to operate in a small observable space while the original plant remains nonlinear and high-dimensional.

## Method details

### 1. Convert a controlled system into a switched autonomous family

The method begins by restricting the admissible control to a finite set of constant values,

\[
\hat U = \{u_0,\ldots,u_{n_c-1}\}.
\]

Each fixed control value defines an autonomous system

\[
G_{u_j}(y) = G(y,u_j).
\]

Thus, the original non-autonomous control problem is replaced by switching among \(n_c\) autonomous systems. In the continuous-time switching-time optimization setting, the system evolves as

\[
\dot y(\cdot,t)=G_{u_j}(y(\cdot,t))
\quad \text{for } t \in [\tau_{\ell-1},\tau_\ell),
\]

where the switching times \(\tau_\ell\) become the decision variables. In the discrete-time MPC setting, with flow maps \(\Phi_{u_j}\), the switched prediction problem is

\[
y_{i+1} = \Phi_{u_{i-s+1}}(y_i),
\qquad u_{i-s+1} \in \hat U.
\]

This transforms the online control decision into choosing a finite switching sequence over a prediction horizon.

### 2. Fit one Koopman reduced model per fixed control value

For each fixed control \(u_j\), the paper approximates a separate Koopman operator. Given an observable map

\[
z=f(y) \in \mathbb R^q,
\]

and a dictionary of basis functions

\[
\psi(z) =
[\psi_1(z),\ldots,\psi_k(z)]^\top,
\]

EDMD is used to approximate the Koopman operator in the lifted feature space. For snapshot pairs \(z_i=f(y_i)\) and \(\tilde z_i=f(\Phi_{u_j}(y_i))\), define

\[
\Psi_Z = [\psi(z_1),\ldots,\psi(z_m)],
\qquad
\Psi_{\tilde Z} = [\psi(\tilde z_1),\ldots,\psi(\tilde z_m)].
\]

The finite-dimensional Koopman approximation satisfies

\[
K^\top = \Psi_{\tilde Z}\Psi_Z^+.
\]

Equivalently, when written through empirical covariance matrices,

\[
K^\top =
(\Psi_{\tilde Z}\Psi_Z^\top)
(\Psi_Z\Psi_Z^\top)^+.
\]

The reduced linear predictor for the lifted observable state is then

\[
\eta_{i+1}=K_{u_j}^\top \eta_i,
\qquad
\eta_i=\psi(f(y_i)).
\]

The key architectural point is that the paper fits \(n_c\) separate linear Koopman predictors, one for each known fixed control value \(u_j\).

### 3. Solve reduced switched control problems online

In the MPC case, the full nonlinear dynamics in the prediction problem are replaced by the K-ROM dynamics:

\[
\min_{u \in \hat U^p}
\sum_{i=s}^{s+p-1} \hat L(\eta_i)
\]

subject to

\[
\eta_{i+1}=K_{u_{i-s+1}}^\top \eta_i.
\]

The online loop is:

1. Observe the current system through \(z_i=f(y_i)\).
2. Lift the observation to \(\eta_i=\psi(z_i)\).
3. Evaluate candidate switching/control sequences over the prediction horizon.
4. Apply the first control from the selected sequence.
5. Repeat at the next sample time.

For small prediction horizons, the paper simply evaluates all possible control sequences. It notes that this brute-force strategy scales poorly because the number of candidates grows exponentially with the number of controls and the horizon length. Dynamic programming and relaxation methods are discussed as alternatives.

### 4. Convergence claim

The paper's theoretical support depends on EDMD convergence and on the objective being expressible through the selected observables. In effect, if the controller's objective can be evaluated in the Koopman feature space, then convergence of the EDMD approximation implies convergence of the K-ROM-based objective values to the full-system objective values for finite switching sequences, in the sense used in the paper.

This is not a general guarantee that arbitrary learned observables or arbitrary finite dictionaries will produce accurate long-horizon controllers. It is a conditional result tied to the chosen observable space, dictionary, data coverage, and EDMD convergence assumptions.

## Results and limitations

The paper evaluates the approach on three examples:

| System | Full-state dimension after discretization | Observable dimension \(q\) | Koopman feature dimension \(k\) | Reported integration speedup |
|---|---:|---:|---:|---:|
| ODE example | 2 | 2 | 6 | about \(20\times\) |
| 1D Burgers equation | 48 | 4 | 35 | about \(100\times\) |
| 2D Navier-Stokes cylinder flow | 22,000 | 8 | 45 | about \(7.5 \cdot 10^4\times\) |

For the ODE switching-time optimization example, the K-ROM objective closely tracks the full objective while being much faster to evaluate. For 1D Burgers, the authors use three fixed controls and low-dimensional point observations, and K-ROM MPC gives control performance close to the full PDE-based MPC. For the Navier-Stokes cylinder-flow example, the controller tracks a desired lift trajectory using only three cylinder rotation values and a low-dimensional observable vector consisting of lift, drag, and six velocity measurements in the wake.

The main limitations are important for comparison with SKAE:

- The control/mode set is fixed in advance. The number \(n_c\) and the identities of the modes \(u_j\) are design choices, not discovered from data.
- Training data are partitioned by known active control. The method knows which autonomous system generated each snapshot pair.
- The approach uses hand-designed observables \(f(y)\) and hand-designed EDMD dictionaries \(\psi\). It does not learn an encoder/decoder representation.
- The route/switching decision is an online control optimization problem, not an inferred basin or support-flow assignment.
- Brute-force MPC over \(\hat U^p\) scales exponentially with horizon length and the number of controls.
- Long-horizon open-loop prediction is limited in the PDE examples; the paper explicitly uses MPC because K-ROM predictions become inaccurate after several steps for those systems.
- Stability of the K-ROM MPC controller is not fully resolved and is listed as future work.
- In the Navier-Stokes example, deviations are attributed partly to control bounds and partly to possible insufficient data richness, which may lead to inaccurate Koopman predictions and incorrect control choices.

## Similarities to the current SKAE staged local \(K_c\) method

The current SKAE staged local \(K_c\) method is:

1. Train one global sparse Koopman autoencoder with shared encoder, shared decoder, and global \(K\).
2. Derive \(C_{\mathrm{stab}}\) routes from learned sparse support-flow fates, without basin labels or known basin counts.
3. Freeze the learned representation and train route-local affine latent maps

\[
T_c(z)=d_c+K_c(z-\bar z_c).
\]

Peitz and Klus are relevant because they provide an early, clear example of using multiple Koopman-style linear models within a switched dynamical/control framework. The following similarities should be acknowledged.

### Multiple linear Koopman predictors

Both approaches ultimately use more than one linear or affine latent/observable evolution model. Peitz and Klus use one Koopman matrix \(K_{u_j}\) per fixed control value. SKAE uses one local affine map \(T_c\) per inferred route \(c\). In both cases, the global nonlinear behavior is represented through simpler local or mode-conditioned linear predictors.

### Low-dimensional prediction instead of full-state simulation

Both approaches aim to avoid modeling the full high-dimensional nonlinear state directly at prediction time. Peitz and Klus predict low-dimensional observables such as sensor values, lift, or drag. SKAE predicts in a learned low-dimensional latent space \(z\), with reconstruction handled by the decoder when needed. Both methods therefore rely on the idea that an appropriately chosen reduced coordinate system can carry the dynamically relevant information.

### Koopman motivation

Both methods draw on the Koopman idea that nonlinear dynamics can become linear when represented through a suitable observable or latent function space. Peitz and Klus realize this with EDMD and a prescribed dictionary \(\psi(f(y))\). SKAE realizes it through a learned encoder and sparse latent dynamics trained as a Koopman autoencoder.

### Switching or routing among dynamics

Peitz and Klus switch among Koopman models during control. SKAE routes among local affine latent maps after discovering route classes from support-flow fates. The common high-level idea is that one linear model may be insufficient for heterogeneous nonlinear dynamics, and that mode-conditional linear models can improve prediction or control.

### Two-stage flavor

Peitz and Klus also separate offline model construction from online use: they first identify Koopman reduced models, then use them inside switching-time optimization or MPC. SKAE similarly separates representation learning, route derivation, and local-map fitting. The stages are different, but both methods distinguish the model-identification phase from the downstream prediction/control phase.

## Differences from the current SKAE staged local \(K_c\) method

The differences are substantial and should be the basis for novelty positioning.

### Exogenous control modes versus endogenous route discovery

In Peitz and Klus, each mode corresponds to a known fixed control input \(u_j\). The mode identity is externally defined before training. The system is switched because the controller chooses among predefined control actions.

In SKAE, each route \(c \in C_{\mathrm{stab}}\) is derived from learned sparse support-flow fates. The route identity is not a known input value, not a control label, and not supplied as supervision. It is an endogenous structure extracted from the trained sparse latent dynamics.

This is the core conceptual distinction: Peitz and Klus perform control over known switched subsystems; SKAE discovers latent dynamical routes from sparse support behavior.

### Known number of modes versus unknown route structure

Peitz and Klus require the finite control set \(\hat U=\{u_0,\ldots,u_{n_c-1}\}\) to be specified. Therefore \(n_c\) is known by construction.

The SKAE setting explicitly avoids assuming the number of basins or routes in advance during training/deployment. Route structure is inferred after learning the global sparse representation. Known basin counts or labels may be used only for benchmark evaluation, not as part of method design.

This matters for paper positioning because SKAE addresses an unsupervised structure-discovery problem that is absent in Peitz and Klus.

### Supervised partitioning by control versus unsupervised support-flow fates

In Peitz and Klus, data for each \(K_{u_j}\) are associated with the active control \(u_j\). Even when data come from a long randomly switched simulation, snapshot pairs are split according to which control was active.

In SKAE, the local maps are trained only after the global representation has been learned and after routes have been inferred from support-flow fates. The partition is not provided by control labels, basin labels, or ground-truth basin counts. This is a different supervision regime.

### Hand-designed observables versus learned sparse latent representation

Peitz and Klus use an observable map \(f(y)\) and EDMD basis \(\psi\) chosen by the practitioner. Their reduced coordinate is

\[
\eta=\psi(f(y)).
\]

SKAE learns the representation itself:

\[
z=E(x),
\qquad
\hat x=D(z),
\]

with shared encoder \(E\), shared decoder \(D\), and a global latent Koopman operator \(K\). Sparsity is not incidental; it is central because the method uses sparse support structure to align basins with supports and then derive route classes. Peitz and Klus do not have an equivalent basin-support alignment objective.

### One global representation first versus separate Koopman models from the start

Peitz and Klus fit separate Koopman matrices for each known subsystem. There is no initial shared global Koopman autoencoder that learns a common representation across all modes.

SKAE first trains a single global sparse Koopman autoencoder with shared encoder/decoder/global \(K\). Only after this global representation is established does the method derive routes and train local affine maps on the frozen representation. This staged design is important: the local maps are not responsible for discovering the representation; they refine dynamics within a representation already structured by sparse support-flow behavior.

### Control switching versus basin-support alignment

Peitz and Klus solve a control problem. Switching is a decision variable chosen to optimize a tracking or regulation objective.

SKAE's primary objective is basin-support alignment: each basin should map to a unique sparse support in latent \(z\), and route-local dynamics should improve modeling of the learned latent flows. SKAE does not primarily optimize a control sequence over a finite action set. The routing is an analysis and modeling step tied to autonomous dynamical organization, not a receding-horizon controller.

### Linear EDMD maps versus centered affine latent maps

Peitz and Klus use linear updates in EDMD feature space:

\[
\eta_{i+1}=K_{u_j}^\top \eta_i.
\]

SKAE's local dynamics are affine and centered:

\[
T_c(z)=d_c+K_c(z-\bar z_c).
\]

This gives each local route a route-specific center \(\bar z_c\), linear component \(K_c\), and offset \(d_c\). That formulation is better matched to local latent flows around route-specific support regions than a single uncentered linear map per externally specified mode.

### Online route selection is different

Peitz and Klus choose the active model by solving a finite-horizon control problem over candidate controls. The selected model is tied to the selected control action.

SKAE derives route membership from the learned support-flow fate. The route is not selected by online MPC search, and its role is not to choose an external actuation. It is a discovered latent dynamical category used for local map fitting and prediction.

### Different evidence target

Peitz and Klus measure control quality and computational acceleration for PDE control. Their evidence target is that K-ROMs enable fast switched control.

SKAE's evidence target is different: the method should show that sparse latent supports align with basins, that support-flow fates induce meaningful routes without labels/counts, and that route-local affine maps improve latent dynamics or downstream prediction while preserving the learned representation.

## Novelty implications for our paper

Peitz and Klus should be cited as relevant prior work for switched Koopman reduced-order models and for using multiple Koopman operators in a control setting. It would be too broad to claim that SKAE is the first method to use multiple Koopman matrices, local linear Koopman models, or switching among Koopman predictors.

However, the paper does not anticipate the central SKAE contribution if our claims are framed precisely. The novelty should be positioned around the following points:

1. **Unsupervised route discovery from sparse latent supports.** Peitz and Klus assume externally defined modes from a finite control set. SKAE derives \(C_{\mathrm{stab}}\) from learned support-flow fates without basin labels or known basin counts.

2. **Basin-support alignment as the organizing principle.** Peitz and Klus do not aim to align basins with sparse latent supports. SKAE's primary scientific claim is that sparse Koopman autoencoding can reveal basin structure through support patterns and support-flow fates.

3. **Global sparse representation before local dynamics.** Peitz and Klus fit one Koopman model per known subsystem. SKAE first learns one global sparse Koopman autoencoder with shared encoder/decoder/global \(K\), then freezes that representation before fitting local affine maps. This staged design separates representation discovery from local dynamical refinement.

4. **No training-time reliance on basin labels/counts.** Peitz and Klus know the mode count and mode identity because these are chosen controls. SKAE should emphasize that, in the intended training/deployment setting, basin labels and basin counts are unavailable.

5. **Local affine latent maps are refinement, not the main discovery mechanism.** To avoid overlap with switched Koopman prior art, the paper should present \(T_c(z)=d_c+K_c(z-\bar z_c)\) as a final route-local refinement enabled by the preceding sparse-support discovery, not as the sole novelty.

The safest novelty statement is therefore:

> Unlike switched Koopman control methods that fit separate Koopman models for known, externally specified modes such as fixed control inputs, SKAE first learns a shared sparse Koopman autoencoder, infers route classes from the fates of learned sparse supports without basin labels or known basin counts, and then fits route-local affine latent dynamics on the frozen representation to improve basin-support-aligned prediction.

This framing acknowledges Peitz and Klus as important prior work while preserving the distinct contribution of SKAE: discovered sparse-support route structure in a learned latent representation, rather than predefined switched control models.
