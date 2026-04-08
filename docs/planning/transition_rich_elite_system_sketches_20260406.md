# Transition-Rich Elite System Sketches

Date: April 6, 2026

## Purpose

This file turns the current elite shortlist from
[transition_rich_system_inventory_20260406.md](/home/mila/l/lia/skae/docs/planning/transition_rich_system_inventory_20260406.md)
into implementation-oriented mechanism sketches. These are still design notes,
not calibrated environments, but they are concrete enough to guide actual
system implementation in the restored worktree.

Status note (April 7 scope freeze):

These sketches are now historical design-provenance notes rather than the live
experiment scope. The active branch experiments are restricted to the fixed
`17`-system shortlist recorded in
[docs/EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md) and
[docs/PAPER_TRACK_STATUS.md](/home/mila/l/lia/skae/docs/PAPER_TRACK_STATUS.md).
Do not use the elite subset below as the current run plan.

Each sketch includes:

- a compact mathematical construction
- the key tunable parameters
- why the mechanism should satisfy the transition-rich criterion if tuned well
- the most likely calibration failure

## Elite Subset

### 1. Triangle Central Gate And Sectors

Let the three core centers be
\[
c_i = R[\cos(2\pi i/3), \sin(2\pi i/3)], \qquad i \in \{0,1,2\}.
\]
Inside each core disk `||x-c_i|| <= r_core`, use a distinct stable linear chart
\[
\dot x = A_i(x-c_i),
\]
with `A_i` Hurwitz and not all conjugate to one another. Inside a shared gate
disk `||x|| <= r_gate`, use
\[
\dot x = -\alpha x + \beta R_{90}x + \sum_i w_i(x)b_i,
\]
where `b_i` points from the origin toward `c_i` and
\[
w_i(x) \propto \exp(\kappa \cos(\theta(x)-\theta_i)).
\]
Key tunables are `R`, `r_core`, `r_gate`, `\kappa`, `\alpha`, `\beta`, and the
eigenstructure of the `A_i`. This should be transition-rich because many
trajectories must leave a local core, pass through the shared gate, and then be
re-routed by angle-dependent weights rather than by nearest-center collapse.
The main failure is a gate that is too weak, which gives trivial local capture,
or too strong, which turns the origin into an accidental attractor.

### 2. Four-Way Crossroads

Put four endpoint sinks at
\[
c_1=(a,0),\quad c_2=(-a,0),\quad c_3=(0,a),\quad c_4=(0,-a).
\]
Use a smooth weighted sink field
\[
\dot x = \sum_i w_i(x)A_i(x-c_i),
\]
with distance-based weights `w_i`, but add axis corridor activations
\[
m_x(x)=\mathbf{1}_{|x_2|<\delta},\qquad m_y(x)=\mathbf{1}_{|x_1|<\delta},
\]
and then a bottleneck term
\[
\dot x \leftarrow \dot x + \gamma[m_x(0,-x_2) + m_y(-x_1,0)]
 + \rho\,\chi_{\|x\|<r_0}R_{90}x.
\]
The axis corridor term forces trajectories into a visible central cross, while
the weak central swirl prevents the intersection from behaving like pure
nearest-quadrant descent. Key tunables are `a`, `\delta`, `\gamma`, `\rho`,
`r_0`, and the local matrices `A_i`. This should be transition-rich because
many starts are forced through a shared bottleneck before final capture. The
failure mode is that the cross becomes either too narrow to matter or too
sticky and turns into a central trap.

### 3. Sector Relay-4

Partition the plane into four angular sectors
\[
S_k = \{x : \theta(x) \in [k\pi/2, (k+1)\pi/2)\},
\]
and place four sinks `c_k` on a square. In each sector, use
\[
\dot x = A_k(x-c_k),
\]
with distinct stable matrices `A_k`. Around the sector boundaries, define a
thin transition strip and add a smooth relay term
\[
\dot x \leftarrow \dot x + \lambda\,\sigma(d_k(x))\,u_{k+1},
\]
where `d_k(x)` is signed distance to the boundary, `u_{k+1}` points into the
next sector, and `\sigma` is a smooth sign or tanh-type transition. Key
tunables are the sink locations, the four `A_k`, strip width, relay strength
`\lambda`, and the steepness of `\sigma`. This should be transition-rich
because the active local chart changes explicitly at sector boundaries, which
forces recurrent chart switching before final capture. The failure mode is
boundary sliding or numerical chatter when the relay is too sharp, or trivial
collapse when it is too weak.

### 4. Lens-Warp Triad

Let `y = \Phi(x)` be a warped coordinate map, for example
\[
\Phi(x) = x + \varepsilon g(\|x\|)R_{90}x,
\]
where `g` is a smooth bump concentrated on one diagonal transport band. In the
warped coordinates define a three-well potential
\[
V(y)= -\sum_{i=1}^3 \exp(-\|y-c_i\|^2/\sigma^2),
\]
with the `c_i` arranged on a triangle, and then evolve in visible coordinates
by
\[
\dot x = -(D\Phi(x))^{-1}\nabla_y V(\Phi(x))
       + \omega(\|x\|)R_{90}\nabla_y V(\Phi(x)).
\]
Key tunables are `\varepsilon`, the bump profile `g`, `\sigma`, the center
spacing, and the swirl gain `\omega`. This should be transition-rich because
the same visible region can send trajectories through different basins
depending on approach direction, even though the underlying endpoint structure
is still simple. The main failure is over-warping, which can turn the
compressed band into a quasi-trapping strip or collapse two basins together.

### 5. Braided Diamond4

Put sinks at the diamond vertices
\[
c_1=(a,0),\quad c_2=(-a,0),\quad c_3=(0,a),\quad c_4=(0,-a),
\]
and start from a smooth weighted sink field
\[
\dot x = -\sum_i w_i(x)A_i(x-c_i),
\qquad
w_i(x)=\frac{\exp(-\|x-c_i\|^2/\sigma^2)}{\sum_j \exp(-\|x-c_j\|^2/\sigma^2)}.
\]
Then add a braided interior term supported on the two diagonals:
\[
\dot x \leftarrow \dot x + \alpha\left[\psi_+(x)R\nabla(x_1+x_2)
 + \psi_-(x)R\nabla(x_1-x_2)\right],
\]
where `\psi_\pm` are narrow Gaussian windows near the origin. The effect is to
create two crossing interior strands that repeatedly re-route trajectories
before capture into a vertex basin. Key tunables are the basin spacing `a`,
overlap width `\sigma`, braid strength `\alpha`, and the anisotropy/rotation of
the `A_i`. The main failure is that the center becomes a corridor attractor, or
that the geometry degenerates into an effectively two-basin split.

### 6. Twin Pinch Bowtie

Let the four sinks be
\[
c_{\pm,\pm}=(\pm a,\pm b),
\]
and define a smooth potential-plus-swirl system
\[
\dot z = -\nabla V(z) + \varepsilon J\nabla V(z),
\qquad
J=\begin{bmatrix}0&-1\\1&0\end{bmatrix}.
\]
Use
\[
V(z)= -\sum_{s\in\{\pm,\pm\}} \exp(-\|z-c_s\|^2/\sigma^2)
 + \lambda_1 \exp\!\left(-\frac{(x^2-a_1^2)^2+y^2}{\rho_1^2}\right)
 + \lambda_2 \exp\!\left(-\frac{(y^2-b_1^2)^2+x^2}{\rho_2^2}\right).
\]
The two ridge terms create a bowtie-shaped web with two pinch points, so
trajectories are funneled through one pinch and then sometimes a second pinch
before final capture. Key tunables are `a`, `b`, `\lambda_1`, `\lambda_2`,
`\rho_1`, `\rho_2`, `\sigma`, and `\varepsilon`. This should be transition-rich
because it gives a real two-stage bottleneck structure rather than one direct
shared corridor. The failure mode is that one pinch dominates and the system
collapses to an effectively two-basin geometry or creates a sticky center.

### 7. Rotating Barrier-4

Again use four sinks at the square corners `(\pm a,\pm a)`, with
\[
\dot z = -\nabla V(z) - \nabla B(z).
\]
Let
\[
V(z)= -\sum_{i=1}^4 \exp(-\|z-c_i\|^2/\sigma^2),
\]
and define an anisotropic barrier band
\[
B(z)=\lambda \exp\!\left(-\frac{(r-r_0)^2}{\rho^2}\right)
\left(\frac{(u(r)^\top z)^2}{\alpha^2} + \frac{(v(r)^\top z)^2}{\beta^2}\right),
\]
where `r=\|z\|`,
\[
u(r)=(\cos\phi(r), \sin\phi(r)),\qquad
v(r)=(-\sin\phi(r), \cos\phi(r)),
\]
and
\[
\phi(r)=\phi_0+\gamma\tanh((r-r_0)/s).
\]
The barrier principal axis rotates with radius, so the preferred exit
direction and effective separatrix shift as the state moves through the band.
Key tunables are barrier height `\lambda`, band center `r_0`, band width
`\rho`, rotation amplitude `\gamma`, sharpness `s`, and anisotropy
`(\alpha,\beta)`. This should be transition-rich because the same attractor
neighborhood can present different effective routes at different radii. The
failure mode is either trivial collapse if the barrier is too weak, or
quasi-trapping / circular wandering if it is too strong.

### 8. Arc DAG4

Place four sinks along an arc, for example
\[
c_i = (r\cos\theta_i, r\sin\theta_i),
\qquad
\theta_1<\theta_2<\theta_3<\theta_4.
\]
Around each sink use a local linear core
\[
\dot x = A_i(x-c_i),
\]
but define four directed transition strips `T_{12}`, `T_{13}`, `T_{24}`,
`T_{34}` and, inside each strip, use a transport field
\[
\dot x = v_{ij} - \beta n_{ij}(n_{ij}\cdot(x-m_{ij})),
\]
where `v_{ij}` points from `i` to `j`, `n_{ij}` is the strip normal, and
`m_{ij}` is the strip centerline. The crucial structural rule is that only
forward transitions `1→2`, `1→3`, `2→4`, and `3→4` are allowed. Key tunables
are arc radius `r`, strip width, contraction `\beta`, and the local core
strengths. This should be transition-rich because many trajectories must pass
through ordered intermediate regions before final capture, which makes the
causal routing structure explicit. The main failure is that the strips become
too weak and the system degenerates into ordinary nearest-well descent, or too
strong and the strips become unintended attractors.

## High-Value Reserve Sketches

These are not in the elite `8`, but they are the strongest immediate reserves.

### Folded Tri Canard

Use the slow-fast system
\[
\epsilon \dot x = y - \left(\frac{x^3}{3}-x\right),\qquad
\dot y = \mu - \nu y - x + \delta \tanh(kx),
\]
with `\epsilon << 1`. The cubic nullcline gives the fold throat and canard
delay, while the tilted `y` dynamics and the `\tanh` term split the plane into
three stable endpoint regions after fold-triggered jumps. Tunables are
`\epsilon`, `\mu`, `\nu`, `\delta`, and `k`. This is a compact reserve slow-fast
system, but the admissible parameter window is narrow and one branch can
disappear or a limit cycle can emerge.

### Triad Fork Graph3

Use a shared central saddle-like decision region near the origin:
\[
\dot x = \mu x_1 - \omega x_2,\qquad
\dot y = -\nu x_2 + \omega x_1,
\]
inside a small disk `D_0`, then route the state into one direct branch leading
to sink `c_1` and one secondary fork cell that later splits between `c_2` and
`c_3`. Tunables are the local saddle parameters `\mu,\nu,\omega`, the fork-cell
location and width, and the sink spacing. This is the cleanest reserve
topological-routing benchmark, but it can degenerate into a one-step split or
turn the fork cell into an accidental fourth basin.

## Suggested implementation order from these sketches

1. Implement one clean control:
   triangle central gate and sectors, or sector relay-4.
2. Implement one explicit routing topology system:
   four-way crossroads, braided diamond4, or arc DAG4.
3. Implement one nontrivial smooth geometry system:
   lens-warp triad.
4. Implement one harder mechanistic stress test:
   rotating barrier-4, or twin pinch bowtie.
5. Keep folded tri canard and triad fork graph3 as immediate reserves if one of
   the elite systems proves too brittle in calibration.
