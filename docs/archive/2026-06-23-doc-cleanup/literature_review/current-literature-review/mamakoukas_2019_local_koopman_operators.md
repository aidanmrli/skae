# Mamakoukas et al. (2019): Local Koopman Operators for Data-Driven Control of Robotic Systems

## Full citation

Giorgos Mamakoukas, Maria Castano, Xiaobo Tan, and Todd Murphey. "Local Koopman Operators for Data-Driven Control of Robotic Systems." *Proceedings of Robotics: Science and Systems*, Freiburg im Breisgau, Germany, June 2019. DOI: `10.15607/RSS.2019.XV.054`.

## Sources read

- Official RSS proceedings entry: <https://www.roboticsproceedings.org/rss15/p54.html>
- Paper PDF hosted by Northwestern Robotics: <https://robotics.northwestern.edu/documents/publications/RSS_2019_May_0.pdf>

This note is based on the paper itself and the official proceedings metadata.

## Problem and motivation

Mamakoukas et al. address real-time feedback control for nonlinear robotic systems. Their motivating setting is underwater robotic locomotion, where the dynamics are nonlinear, affected by fluid interactions, and expensive to control with fully nonlinear methods at deployment time. Koopman lifting is attractive because it seeks a representation in which nonlinear dynamics evolve linearly in observable space, after which standard linear control tools such as LQR can be used.

The paper's central practical problem is basis-function selection for finite-dimensional Koopman approximations. The exact Koopman operator is infinite-dimensional except in special cases, and finite invariant subspaces are not generally available. The authors argue that, for systems with known nonlinear model structure but possibly unknown coefficients, higher-order time derivatives of the model terms provide a principled observable dictionary. A data-driven least-squares fit then estimates a finite Koopman operator over those observables.

The goal is not multibasin regime discovery, representation learning, or label-free partitioning. The goal is to obtain a finite lifted linear model that supports fast LQR feedback on a robotic fish, with LQR gains computed once and applied online at low cost.

## Method details

### Koopman finite-dimensional approximation

For a state `s` and observables `\Psi(s) = [\psi_1(s), ..., \psi_w(s)]`, the Koopman form is written as

```tex
\frac{d}{dt}\Psi(s) = K\Psi(s),
\qquad
\Psi(s_{k+1}) = K_d \Psi(s_k).
```

For controlled data, observables include both state and control terms:

```tex
\Psi(s_k, u_k).
```

Given `P` measurements `(s_k, u_k, s_{k+1}, u_{k+1})`, the discrete finite Koopman operator is fit by least squares:

```tex
\tilde K_d^*
= \arg\min_{\tilde K_d}
\sum_{k=0}^{P-1}
\frac{1}{2}
\left\|
\Psi(s_{k+1}, u_{k+1})
- \tilde K_d \Psi(s_k, u_k)
\right\|_2^2.
```

The closed-form solution is

```tex
\tilde K_d^* = A G^\dagger,
```

with

```tex
A =
\frac{1}{P}\sum_{k=0}^{P-1}
\Psi(s_{k+1}, u_{k+1})\Psi(s_k, u_k)^T,
\qquad
G =
\frac{1}{P}\sum_{k=0}^{P-1}
\Psi(s_k, u_k)\Psi(s_k, u_k)^T.
```

The continuous-time operator is obtained from the discrete operator by

```tex
\tilde K = \log(\tilde K_d) / \delta t.
```

### Observable synthesis from higher-order derivatives

The paper's main methodological idea is to populate the observable dictionary using higher-order derivatives of known nonlinear dynamics. The motivation is Taylor expansion. For a nonlinear term `g(t)`,

```tex
g(t)
\approx
g(t_0)
+ g'(t_0)\delta t
+ \frac{1}{2}g''(t_0)\delta t^2
+ \cdots
+ \frac{1}{n!}g^{(n)}(t_0)\delta t^n.
```

Stacking derivative terms gives an approximate local-in-time linear propagation:

```tex
\begin{bmatrix}
g(t) \\
g'(t) \\
g''(t) \\
\vdots \\
g^{(n)}(t)
\end{bmatrix}
\approx
\tilde K_d
\begin{bmatrix}
g(t_0) \\
g'(t_0) \\
g''(t_0) \\
\vdots \\
g^{(n)}(t_0)
\end{bmatrix}.
```

The analytic Taylor matrix is not used as the final operator because the derivative stack generally does not close after finite truncation. Instead, derivatives are used to choose candidate observables, and the operator is fit from data. This is why "local" in this paper primarily means local in time through a truncated Taylor/derivative argument, not a routed family of local models over regions of state space.

The authors emphasize that their procedure is not intended for completely unknown dynamics. It assumes a model form is available so that derivative terms can be generated. The coefficients of those terms need not be known, since the Koopman matrix is learned from data.

### Koopman-LQR controller

The authors split controlled observables into state-only and control-dependent components:

```tex
\Psi(s,u)
=
\begin{bmatrix}
\Psi_s(s) \\
\Psi_{s,u}(s,u)
\end{bmatrix}.
```

The approximate Koopman dynamics are partitioned as

```tex
\frac{d}{dt}
\begin{bmatrix}
\Psi_s(s) \\
\Psi_{s,u}(s,u)
\end{bmatrix}
\approx
\begin{bmatrix}
\tilde K_s & \tilde K_{s,u} \\
\tilde K_{u,s} & \tilde K_{u,u}
\end{bmatrix}
\begin{bmatrix}
\Psi_s(s) \\
\Psi_{s,u}(s,u)
\end{bmatrix}.
```

For speed, they choose

```tex
\Psi_{s,u}(s,u) = u,
```

which yields a fixed affine lifted dynamics model:

```tex
\frac{d}{dt}\Psi_s(s)
\approx
\tilde K_s \Psi_s(s) + \tilde K_{s,u}u.
```

The lifted infinite-horizon LQR objective is

```tex
J_{\tilde K}
=
\int_0^\infty
(\Psi_s(s)-\Psi_s(s_{\mathrm{des}}))^T
Q_{\tilde K}
(\Psi_s(s)-\Psi_s(s_{\mathrm{des}}))
+ u^T R u \, dt,
```

with feedback

```tex
u =
-K_{\mathrm{LQR}}
(\Psi(s)-\Psi(s_{\mathrm{des}})).
```

The main computational benefit is that the LQR gains are computed once. Online control then only requires evaluating observables and applying the fixed gain matrix.

### Experimental system

The case study is a tail-actuated robotic fish with state

```tex
s = [x, y, \psi, v_x, v_y, \omega]^T,
```

where `(x,y)` are world-frame position, `\psi` is orientation, `v_x,v_y` are body-frame velocities, and `\omega` is angular velocity. The tail actuation is parameterized by amplitude, bias, and frequency:

```tex
\alpha(t) = \alpha_0 + \alpha_a \sin(\omega_a t).
```

In simulation, the authors populate the observable dictionary with states, control inputs, and first-order derivatives of terms appearing in the fish dynamics. This yields `\Psi_x(s) \in R^{60}`: the state and input coordinates plus 52 additional scalar functions. They fit the discrete Koopman operator using `P = 3000` sampled transitions with `\delta t = 0.005 s`, convert it to continuous time, and compute infinite-horizon LQR gains.

## Results and limitations

### Reported results

- In simulation, the Koopman-LQR controller tracks a figure-eight target trajectory for the robotic fish using fixed LQR gains computed from the learned Koopman representation.
- In physical experiments, the authors collect 22 robotic fish runs: two trials for each of 11 amplitude/bias actuation settings, with tail frequency fixed at `\omega_a = 2\pi` rad/s.
- State measurements are obtained from an overhead camera at roughly 4 Hz, velocities are estimated with Kalman filtering, and the data are interpolated to `\delta t = 0.005 s` before fitting the Koopman operator.
- The learned linear Koopman model follows the measured experimental trajectories reasonably for at least about five seconds in the shown comparisons.
- The closed-loop Koopman-LQR controller is evaluated on line, arc, and circle tracking tasks. Feedback is applied at 1 Hz, limited by image-processing speed. The paper reports that performance is comparable to backstepping control on the same robotic fish.

### Limitations

- The observable construction requires known nonlinear model structure. The method is not a fully data-driven representation learner for unknown systems.
- The paper does not learn an encoder or decoder. The lifted coordinates are hand-constructed from state variables, control variables, and derivative-generated model terms.
- The approach fits one Koopman model for the chosen observable dictionary. It does not discover multiple regimes, basins, local charts, or a routing function.
- The word "local" refers to the Taylor/derivative argument and finite-time approximation, not to local operators selected by state-space regions or learned supports.
- The choice `\Psi_{s,u}(s,u)=u` is made for online speed, but it discards richer state-control interaction observables. The authors explicitly note that this can reduce approximation accuracy.
- The empirical evidence is concentrated on one robotic platform and is largely qualitative in the paper figures. The paper does not provide a broad benchmark suite or statistical comparison across many systems.
- The authors list formal guarantees for derivative-basis optimality and finite-order accuracy as future work.
- The paper acknowledges the standard Koopman limitation that finite-dimensional invariant subspaces are unavailable in general and notes prior results that systems with multiple fixed points lack finite-dimensional Koopman closure. This matters for SKAE because our target systems are explicitly multibasin.

## Similarities to the current SKAE staged local `K_c` method

Both methods use the Koopman idea to turn nonlinear dynamics into linear or affine evolution in a lifted space. In Mamakoukas et al., the lifted space is a hand-designed observable dictionary. In the SKAE staged method, the lifted space is the learned sparse latent code `z` produced by an encoder. In both cases, the downstream dynamics are intentionally simple: a linear operator in the lifted coordinates, or a small family of affine operators in SKAE.

Both methods separate representation/dictionary choice from operator use. Mamakoukas et al. first choose derivative-informed observables and then fit a finite operator. SKAE first trains a global sparse Koopman autoencoder with shared encoder, decoder, and global `K`, then uses the learned sparse support-flow structure to define routes and train local maps after freezing the representation.

Both methods are motivated by the practical value of simple lifted dynamics. Mamakoukas et al. use the lifted linear model for fast LQR control. SKAE uses learned sparse supports and local affine latent maps for interpretable multibasin forecasting and regime-conditioned dynamics. In both cases, the method trades exact nonlinear modeling for a computationally manageable lifted approximation.

Both methods are relevant to the finite-closure problem in Koopman modeling. Mamakoukas et al. use derivative stacks to approximate a Koopman-invariant subspace when exact closure is unavailable. SKAE does not claim exact global closure either; instead, it asks whether sparse learned support fates can carve multibasin dynamics into useful route-conditioned latent affine maps.

Both methods can be described as "local" only in a broad Koopman approximation sense. Mamakoukas et al. obtain a local-in-time approximation motivated by Taylor expansion. SKAE obtains route-local latent maps conditioned on support-flow fate. The shared word is real but the mechanism is different.

## Differences from the current SKAE staged local `K_c` method

The most important difference is the representation. Mamakoukas et al. use a physics-informed observable dictionary built from known nonlinear model terms and their derivatives. SKAE learns the representation from trajectory data using a sparse Koopman autoencoder:

```tex
z_t = \mathrm{Enc}(x_t),
\qquad
\hat x_t = \mathrm{Dec}(z_t),
\qquad
z_{t+1} \approx K z_t
```

with one shared encoder, one shared decoder, and one shared global transition `K` in the first stage.

The second major difference is routing. Mamakoukas et al. do not route states to different Koopman operators. There is no learned regime variable, no support graph, no basin-fate object, and no local model selection during rollout. SKAE derives routes from learned sparse support-flow fates:

```tex
m_{\mathrm{abs}}(x_t)
\rightarrow
u_t
\rightarrow
c = C_{\mathrm{stab}}(u_t),
```

where exact or high-resolution sparse support masks define base support-family labels, support transitions define an empirical support-flow graph, recurrent support-flow fates define `C_{\mathrm{stab}}`, and `c` selects the local affine latent map.

The assumptions about labels and counts are also different. Mamakoukas et al. do not address basin labels or basin counts because their problem is not multibasin discovery. SKAE explicitly assumes that at training and deployment time we do not know the number of basins and do not know which trajectories belong to which basin. `C_{\mathrm{stab}}` is therefore constructed from the model's learned support trajectories, not from ground-truth basin assignments.

The local operator form differs. Mamakoukas et al. fit a single lifted operator and then extract fixed state/control blocks for LQR:

```tex
\frac{d}{dt}\Psi_s(s)
\approx
\tilde K_s \Psi_s(s) + \tilde K_{s,u}u.
```

SKAE freezes the learned representation and trains route-specific affine latent maps:

```tex
T_c(z) = d_c + K_c(z-\bar z_c),
```

where `c` is selected from `C_{\mathrm{stab}}`, `\bar z_c` is a route center, `d_c` is a learned affine target/intercept term, and `K_c` is a route-local latent transition. This is closer to a learned mixture of latent affine charts than to the derivative-observable Koopman-LQR construction in Mamakoukas et al.

The training procedures are different. Mamakoukas et al. fit `\tilde K_d` by closed-form least squares over a fixed dictionary. SKAE uses neural training for the global sparse Koopman autoencoder, then a second-stage local-map procedure with the encoder, decoder, and global `K` frozen. In SKAE, the first-stage representation is responsible for exposing support structure; the second stage only learns local affine dynamics conditional on that support-derived route.

The role of supervision is different. Mamakoukas et al. require structural knowledge of the dynamics, but not basin labels. SKAE does not require model equations, basin labels, basin counts, or trajectory-to-basin assignments for training/deployment. Benchmark basin labels may be used only after training for evaluation of separability or basin-support alignment.

The scientific target is different. Mamakoukas et al. target real-time robotic control and compare Koopman-LQR to backstepping. SKAE targets basin-support alignment and label-free support-routed dynamics in multibasin systems. Control synthesis is not the central claim of the staged local `K_c` result.

## Novelty implications for the SKAE paper

Mamakoukas et al. should be cited as relevant prior art for Koopman-based robotic control, derivative-informed observable construction, finite least-squares Koopman fitting, and Koopman-LQR with fixed online gains. It is especially relevant because the title uses "Local Koopman Operators" and because the paper explicitly discusses finite-dimensional Koopman closure limitations.

It is not a close anticipation of the staged SKAE local `K_c` method. The paper does not learn sparse supports, does not construct a support-flow graph, does not derive `C_{\mathrm{stab}}` routes, does not train route-specific local affine maps after freezing a global autoencoder, and does not address the training/deployment constraint that basin labels and basin counts are unknown. Its locality is Taylor/derivative locality in time, whereas SKAE's locality is support-fate-conditioned latent dynamics.

The strongest novelty-safe framing for our paper is:

- Prior Koopman control work, including Mamakoukas et al., shows that carefully chosen lifted coordinates can make nonlinear robotic dynamics usable by linear control tools.
- Our contribution is not merely "using local Koopman operators." It is learning a sparse Koopman representation whose active-coordinate support trajectories induce label-free regime routes.
- The staged local `K_c` experiment then tests whether those learned support-fate routes can select useful local affine latent dynamics:

```tex
\mathrm{Enc}, \mathrm{Dec}, K
\quad \text{trained globally first;}
\qquad
C_{\mathrm{stab}}
\quad \text{derived from learned support-flow fates;}
\qquad
T_c(z)=d_c+K_c(z-\bar z_c)
\quad \text{trained after freezing the representation.}
```

The main reviewer risk is terminology. A reader may initially conflate Mamakoukas et al.'s "local Koopman operators" with our staged local `K_c` maps. The related-work text should avoid claiming novelty for "local Koopman" generically. It should claim novelty for support-flow-routed local affine latent dynamics learned without basin labels or known basin counts, on top of a globally trained sparse Koopman autoencoder.

The paper also helps motivate why our multibasin setting is nontrivial. Mamakoukas et al. rely on derivative-informed finite approximations but acknowledge the broader finite-closure problem, including the difficulty of multiple fixed points. SKAE can be positioned as a complementary empirical strategy for multibasin systems: instead of seeking one exact finite invariant subspace, learn sparse supports whose flow fates provide a label-free routing object for local latent affine approximations.
