# Ju et al. (2026): Koopman MPC for Nonlinear Switched Systems

## Full Citation

Xuqian Ju, Qing Sun, Dario Dennstädt, Karl Worthmann, Dajun Du, and Minrui Fei. "Model predictive control for nonlinear switched systems based on Koopman theory." *Systems & Control Letters* 209, article 106340, 2026. DOI: [10.1016/j.sysconle.2025.106340](https://doi.org/10.1016/j.sysconle.2025.106340).

The article reports: received March 26, 2025; revised December 1, 2025; accepted December 28, 2025; available online January 15, 2026.

## Source Consulted

I read the article from the local PDF:

`docs/Ju et al. - 2026 - Model predictive control for nonlinear switched systems based on Koopman theory.pdf`

The paper itself lists the DOI above and the journal page `www.elsevier.com/locate/sysconle`. This note is based on that local PDF.

## Problem And Motivation

Ju et al. address control of discrete-time nonlinear switched systems, where each mode has its own nonlinear dynamics and the system may switch among modes according to either a known schedule or a sequence chosen by the controller. Their motivating concern is that standard Koopman MPC methods typically fit one global lifted linear model and therefore ignore mode-dependent dynamics. In switched systems this can produce poor prediction because abrupt changes in the active subsystem violate the single-model assumption.

The paper positions switching as both a modeling difficulty and a control opportunity. In some applications, the switching sequence is exogenous or rule-based; in others, it can be optimized jointly with continuous controls to improve performance. The authors aim to keep the online MPC problem close to linear-quadratic MPC by learning linear predictors in lifted coordinates, while also accounting for switching explicitly.

## Method Details

The base controlled nonlinear system is written as

```text
x_{k+1} = f(x_k, u_k),
```

with compact polyhedral state and input constraints `x in X`, `u in U`. For switched systems, the mode-dependent dynamics are

```text
x_{k+1} = f_{sigma(k)}(x_k, u_k),      sigma(k) in R = {1, ..., N_s}.
```

The paper assumes the switching times are either known a priori or treated as decision variables. This is central: mode identity is part of the system description, and training data are available per mode.

### Koopman/EDMDc Predictor

The authors use a shared observable dictionary

```text
Psi(x) = [psi_1(x), ..., psi_N(x)]^T
```

and construct one EDMDc-style lifted linear predictor per mode:

```text
z_{k+1} = K_sigma z_k + B_sigma u_k
x_hat_k = C z_k
z_k = Psi(x_k).
```

For each mode, data matrices are formed from mode-specific snapshots. The paper defines lifted state blocks `X_lift`, successor lifted blocks `Y_lift`, and control blocks `U`; then fits

```text
[K_sigma | B_sigma] =
    argmin_{K, B} || K X_lift + B U - Y_lift ||_F^2,
```

with a shared linear decoder

```text
C = argmin_C || C X_lift - X ||_F^2.
```

The local predictors are therefore mode-specific in dynamics but share the same lifted coordinate system and decoder.

### Dictionary Choice And Neural Observables

The default dictionary in the experiments uses 100 thin-plate spline radial basis functions of the form

```text
psi_i(x) = ||x - c_i||^2 log ||x - c_i||,
```

with centers sampled uniformly. The paper also discusses learned observables using a compact multilayer perceptron with PReLU activations. The neural dictionary is trained by a residual-sum-of-squares objective that combines reconstruction and Koopman-invariance terms:

```text
L_RSS(Psi; C, K, B)
  = sum_i || x_i - C Psi(x_i) ||^2
    + lambda sum_sigma sum_i
        || K_sigma Psi(x_i) + B_sigma u_i - Psi(f_sigma(x_i, u_i)) ||^2.
```

The authors emphasize the usual EDMD tradeoff: richer dictionaries improve fidelity but increase regression size, lifted dimension, and MPC cost.

### S-KMPC Optimization

The switched Koopman MPC problem uses the lifted predictors over a finite horizon `N_h`. When switching is optimized, the mode sequence is a decision variable:

```text
minimize over u_i, sigma_i:
    terminal_cost(z_{k+N_h}) +
    sum_{i=k}^{k+N_h-1} (z_i^T Q z_i + u_i^T R u_i)

subject to:
    z_k = Psi(x_k)
    z_{i+1} = K_{sigma_i} z_i + B_{sigma_i} u_i
    C z_i in X
    u_i in U
    z_{k+N_h} in X_f.
```

If switching is known, the paper adds the constraint that the optimized sequence equals the prescribed switching signal over the horizon.

The terminal set is defined through a quadratic Lyapunov function in lifted coordinates:

```text
X_f = { z : V(z) <= c },
V(z) = z^T P z,     P = P^T > 0.
```

For stability, the paper assumes there are mode-dependent terminal feedback matrices `F_sigma` such that the closed-loop lifted matrices

```text
A_sigma = K_sigma + B_sigma F_sigma
```

share a common quadratic Lyapunov function:

```text
A_sigma^T P A_sigma - P < 0      for all modes sigma.
```

Under the exact finite-dimensional Koopman lifting assumption, the paper proves recursive feasibility and asymptotic stability for the nominal regulation problem. The authors explicitly note that the common quadratic Lyapunov function requirement is strong and sufficient but not necessary.

### Model Mismatch

The exact Koopman setting is relaxed in discussion by introducing a proportional one-step lifted error:

```text
Psi(f_sigma(x, u)) = K_sigma Psi(x) + B_sigma u + e_sigma(x, u),
||e_sigma(x, u)|| <= c_x ||x|| + c_u ||u||.
```

The paper does not fully solve the robust case. It points to constraint tightening and robust MPC extensions as future work and notes that verifying such error bounds for linear Koopman representations remains open.

## Results And Limitations

### Van der Pol Switched Prediction

The first benchmark is a forced Van der Pol oscillator with two modes and known switching sequences. The authors compare a single-mode EDMDc predictor fit to the full dataset against the switched Koopman predictor.

Reported average RMSE over 200 random initial conditions:

| Setting | Classic single-mode Koopman | Switched Koopman |
|---|---:|---:|
| Switching sequence 1 | 115.74% | 20.87% |
| Switching sequence 2 | 155.49% | 21.58% |

With an MLP dictionary, the paper reports MRMSE of 1.65% on this benchmark.

### Switched DC Motor

The second benchmark is an armature-controlled DC motor with two modes. The authors again compare single-mode and switched predictors, including a neural dictionary variant.

Reported prediction errors over 200 random initializations:

| Predictor | MRMSE | RMSE |
|---|---:|---:|
| Single-mode, sequence 1 | 58.98% | 118.0% |
| Single-mode, sequence 2 | 57.98% | 116.0% |
| Switched, sequence 1 | 33.58% | 85.2% |
| Switched, sequence 2 | 22.60% | 55.8% |
| Switched + MLP, sequence 2 | 6.31% | not reported |

In closed-loop MPC, the paper reports that S-KMPC respects constraints and matches the classic KMPC tracking error while reducing online solve time from 21.8 s to 10.1 s per step. The MLP basis required 209 s offline training and gave the most accurate predictor.

### Switching-Sequence Optimization

The third demonstration uses a two-mode Duffing oscillator. At each sampling instant, the controller generates a small library of admissible short mode prefixes, solves convex S-KMPC QPs for candidate completions, adds a switching penalty proportional to the number of mode changes, and applies the first mode/control from the best prefix. The reported behavior is accurate tracking with bounded inputs and sparse switching. The paper explicitly states that it does not provide a formal stability proof for this time-varying tracking setup with switching penalties; the theory is for nominal regulation.

### Limitations

The main limitations are:

- The method assumes a finite set of modes is known.
- Training relies on mode-specific data, so mode identity or switching-time information is part of the identification setup.
- The strongest feasibility/stability guarantees require an exact finite-dimensional Koopman lifting and a common quadratic Lyapunov function for all closed-loop lifted modes.
- The common Lyapunov condition is restrictive and may hold only in special cases.
- Robustness to approximation error is discussed but not fully proved in the main S-KMPC result.
- Benchmarks are low-dimensional and noise-free by design, which isolates lifting effects but does not establish scalability to high-dimensional learned representations.

## Similarities To The SKAE Staged Local `K_c` Method

The closest methodological similarity is that both approaches use a shared lifted/latent coordinate system with multiple local linear dynamics maps.

In Ju et al., the shared object is the observable dictionary `Psi` and decoder `C`; the mode-specific objects are `K_sigma` and `B_sigma`. In the current SKAE staged method, the shared object is the learned sparse Koopman autoencoder representation, including shared encoder, decoder, and global `K`; the local objects are affine latent maps

```text
T_c(z) = d_c + K_c (z - zbar_c).
```

Both methods are motivated by the inadequacy of a single global Koopman predictor when the underlying dynamics are heterogeneous. Ju et al. show this explicitly: a single-mode EDMDc predictor fit across switched data performs much worse than mode-specific predictors. This supports the broad intuition behind staged local `K_c`: after learning a common representation, local dynamics can improve prediction in regions/routes where a single map is too coarse.

Both approaches also separate representation from local dynamics to some degree. Ju et al. use one common dictionary and fit separate mode-wise linear maps. SKAE first trains one global sparse Koopman autoencoder, then freezes the representation before fitting local affine latent maps. This is conceptually aligned with using a common coordinate system to make local linear modeling meaningful.

A third similarity is that both methods can be described as routing into local linear dynamics. Ju et al. route by the active switched-system mode `sigma`. SKAE routes by the learned sparse support-flow fate `C_stab`. In both cases, prediction/control after routing uses a linear or affine latent update chosen from a finite collection.

Finally, both papers care about preserving a useful linear structure after lifting. Ju et al. need it for convex QP MPC and common-Lyapunov analysis. SKAE needs it for interpretable sparse latent dynamics and basin-support alignment.

## Differences From The SKAE Staged Local `K_c` Method

The most important difference is supervision and problem setup. Ju et al. work with switched systems whose finite mode set is part of the model. The training data are partitioned by mode, and the controller either receives the switching sequence or optimizes over known mode labels. SKAE is designed for a training/deployment setting where the number of basins is not known in advance and trajectories are not labeled by basin.

The routing mechanism is therefore fundamentally different. In Ju et al., the route is the physical or hybrid-system mode `sigma`; it is known, prescribed, or explicitly searched. In SKAE, `C_stab` routes are derived from learned sparse support-flow fates. The route is an emergent object from the learned representation, not an externally supplied mode label.

The representation learning objective is also different. Ju et al. learn or choose observables to support mode-wise Koopman prediction and MPC. Their neural objective combines reconstruction with Koopman-invariance residuals for known modes. SKAE prioritizes basin-support alignment: each basin should map to a unique sparse support in latent `z`. Local `K_c` training is a later refinement after the global sparse representation and support-flow routes have been established.

The order of training differs:

1. Ju et al.: choose/learn a shared dictionary, then fit each mode-specific EDMDc predictor using mode-specific data.
2. SKAE: train one global sparse Koopman autoencoder with shared encoder/decoder/global `K`; derive `C_stab` routes from learned support-flow fates without basin labels/counts; freeze the representation; train local affine maps `T_c`.

The local maps also have different semantics. Ju et al.'s `K_sigma, B_sigma` approximate controlled dynamics under known subsystem modes. SKAE's `K_c` maps approximate route-conditioned latent dynamics around route centers `zbar_c`, with an affine offset `d_c`; `c` indexes a discovered support-flow fate rather than a known subsystem.

The control emphasis differs. Ju et al. are primarily an MPC paper: the method is judged by prediction, constraint satisfaction, recursive feasibility, stability, and solve time. SKAE is primarily a representation and dynamical-structure paper: the central claim is not convex MPC, but discovering sparse basin-aligned supports and then improving local latent dynamics without using labels or known basin counts.

The assumptions behind the theoretical guarantees also diverge. Ju et al. require an exact finite-dimensional Koopman lifting for the nominal theorem and a common quadratic Lyapunov function across modes. SKAE should not inherit those guarantees unless it explicitly proves analogous conditions for learned sparse latent supports and local affine maps. Conversely, SKAE addresses unsupervised basin discovery, which is outside Ju et al.'s main setup.

## Novelty Implications For The SKAE Paper

Ju et al. make it unsafe to claim novelty merely as "a Koopman model with multiple local/switching linear maps in one lifted space." That idea is clearly present here: they use a shared observable dictionary and mode-specific linear predictors for switched nonlinear systems.

However, their paper does not remove the core novelty of the staged SKAE method if our claim is framed correctly. The defensible novelty is not multiple `K_c` maps by itself. It is the pipeline:

```text
global sparse Koopman autoencoder
  -> support-flow fate discovery without basin labels/counts
  -> C_stab route construction
  -> frozen-representation local affine latent dynamics
```

The strongest distinction is the absence of supervised mode/basin information. Ju et al. know the mode set and use mode-specific data. SKAE derives route structure from sparse latent support dynamics and is intended for settings where the number of basins and trajectory-to-basin assignments are unknown at training/deployment time.

For paper positioning, Ju et al. should be cited as closely related work on switched Koopman MPC and mode-specific Koopman predictors. The contrast should say that S-KMPC assumes an explicit switched-system mode structure and uses known or optimized switching sequences, while SKAE discovers route structure from sparse latent supports and only then fits local affine dynamics.

The local `K_c` stage should be presented as a refinement enabled by the learned basin-support alignment, not as the primary novelty. The novelty claim is strongest if the paper foregrounds:

- sparse latent support as the basin-identifying object;
- unsupervised route discovery from support-flow fates;
- no training-time reliance on basin labels or fixed basin counts;
- staged separation between global representation learning and local affine dynamics;
- evidence that discovered supports align with basins and improve local prediction/control-relevant modeling.

In short: Ju et al. overlap with SKAE at the broad "shared lifted space plus multiple linear maps" level, but they operate in a supervised switched-system/control setting. SKAE remains novel if framed as unsupervised basin-support alignment with post hoc local latent linearization, rather than as another switched Koopman MPC method.
