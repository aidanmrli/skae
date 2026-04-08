# Basin partition experiment design note

Status note:

- Treat this file as the current planning source of truth for ablation design
  choices tied to items `3` and `4` of
  [docs/planning/transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md).
- It records candidate axes to test systematically on the interpretability
  branch.
- Replace design assumptions here with experiment-backed conclusions once the
  relevant runs have been completed and written back into
  [docs/planning/transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md),
  [docs/EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md), and
  [docs/PAPER_TRACK_STATUS.md](/home/mila/l/lia/skae/docs/PAPER_TRACK_STATUS.md).

Let me restate the setup in the narrow form you want.

You want **one** sparse LISTA-style encoder (E), **one** linear latent transition (K), a linear dictionary decoder (D), and **periodic or event-triggered re-encoding** so that the effective forecast map is nonlinear even though the latent propagator is linear:
[
z_t = E(x_t),\qquad \tilde z_{t+1}=K z_t,\qquad \hat x_{t+1}=D(\tilde z_{t+1}),
]
and then
[
z_{t+1}=
\begin{cases}
\tilde z_{t+1}, & r_t=0[2mm]
E(\hat x_{t+1}), & r_t=1,
\end{cases}
]
where (r_t) is the reset decision. With (r_t\equiv 0), this is a standard Koopman autoencoder rollout. With resets, the effective state-space map becomes nonlinear and can behave like a sequence of local linear charts. That matters here because periodic reencoding is specifically motivated as a middle ground between global latent linearity and reencoding every step, and multiple fixed points are one of the failure modes Fathi et al. call out for no-reencoding rollouts. Separately, Pan and Duraisamy explain why multiple disjoint invariant sets are problematic for a single globally continuous finite-dimensional lifting with linear reconstruction. ([ar5iv][1])

One important nuance for your 2D autonomous multistable toys: a noiseless trajectory does **not** physically hop between basins after initialization. So in these experiments, resets are mostly for **chart correction**, **support reselection**, and **preventing latent drift**, not for literal basin switching along a single trajectory. That actually makes hard block structure more plausible here than it would be in a noisy or controlled setting.

## 1) Koopman matrices: what exactly to test

I would test three matrix families, but I would make them explicit in a way that lets you study identifiability.

### A. Dense (K)

This is the unconstrained baseline:
[
K\in\mathbb R^{p\times p}.
]

Nothing in the architecture says what a basin is. If basin structure emerges, it must emerge through the sparse encoder and the reset map (E\circ D). This is the most flexible family and usually the hardest one to interpret.

What to test:

* Dense, no structural penalty.
* Dense plus a stability penalty on the spectrum or operator norm.
* Dense with no resets, fixed-period resets, and event-triggered resets.

What I would expect:

* Best raw flexibility.
* Weakest basin identifiability.
* Re-encoding must do almost all the “local chart switching.”
* A dense (K) may forecast well while giving you very little clean structure in supports or columns/rows of (K).

### B. Hard block-diagonal (K)

Choose a latent partition (G_1,\dots,G_B), optionally with a shared block (G_0). After a fixed permutation (P),
[
K = P^\top \operatorname{blkdiag}(K_0,K_1,\dots,K_B)P.
]

If you do not want a shared block, drop (K_0). If you do want one, (K_0) is for common modes and (K_1,\dots,K_B) are local/basin blocks.

For autonomous multistable systems, I would define (B) relative to the number of **stable attractors**, not the total number of fixed points including saddles.

What to test:

* (B\in{n_a-1,n_a,n_a+1,2n_a}), where (n_a) is the number of stable attractors.
* With and without a shared block.
* Equal block sizes first, then one shared-small + basin-specific-large split.

What I would expect:

* Best identifiability if the encoder aligns groups with basins.
* Very competitive forecasting on your toy systems, because each trajectory should mostly stay in one basin.
* Possible underfitting if there are genuinely shared transient modes near saddles/separatrices or if (B) is mis-specified.

### C. Soft block-sparse (K)

Use the same latent partition, but do **not** zero out off-block entries. Instead write
[
K = K_{\text{in}} + K_{\text{off}},
\qquad
K_{\text{in}} = M\odot \Theta,\qquad
K_{\text{off}}=(1-M)\odot \Theta,
]
where (M) is the within-block mask and (\Theta) is free. Then penalize (K_{\text{off}}) rather than forcing it to zero.

This is the one-matrix version I would trust most if you want both forecasting and implicit regime structure.

What to test:

* Same (B) sweep as above.
* A sweep over (\lambda_{\text{block}}) from (0) to “almost hard block diagonal.”
* A shared-block variant.

What I would expect:

* Best trade-off between forecast accuracy and identifiability.
* Enough flexibility for shared/separatrix dynamics.
* Cleaner supports than dense (K).
* Less brittle than hard block diagonal when the encoder is imperfect.

### What to measure for these matrix families

For block models, define group activity
[
a_b(z)=|z_{G_b}|_2.
]
This lets you inspect whether one group dominates, whether that dominance stays stable along a trajectory, and whether dominant groups align with true basins.

For identifiability, I would explicitly compare:

* Forecast error vs horizon.
* Dominant-group purity vs true attractor label.
* Off-block energy ratio
  [
  \frac{|(1-M)\odot K|_1}{|K|_1}.
  ]
* Whether different basins map to different dominant groups.
* Whether a single trajectory keeps a stable dominant group.

If you want one hypothesis up front: for your autonomous toys, I would expect **hard block-diagonal** to be strongest for interpretability, and **soft block-sparse** to be the best overall compromise.

## 2) Loss terms: what to include, why, and how to combine them

I would define all losses in terms of the rollout operator that matches inference. Let
[
R_0^\pi(z)=z,
]
and recursively
[
R_{h+1}^\pi(z)=
\begin{cases}
K,R_h^\pi(z), & \pi_h=0,[2mm]
E!\left(D!\left(K,R_h^\pi(z)\right)\right), & \pi_h=1,
\end{cases}
]
where (\pi_h) is the reset policy, periodic or event-triggered.

That way your training loss reflects the actual rollout mechanism you intend to use at test time. This matters because Fathi et al. distinguish reencoding from teacher forcing: reencoding uses the model’s own prediction, not ground truth. tcKAE adds a different but compatible idea: latent temporal consistency across different time origins. Standard KAE variants also commonly use reconstruction, prediction, and linearity/alignment terms, and some recent work adds orthogonality-like stabilization. ([ar5iv][1])

### Core losses

**Reconstruction**

Purpose: keep the autoencoder honest. Without this, sparse supports are hard to interpret.

**Latent linearity / alignment**

Purpose: enforce that encoded states evolve approximately linearly one step at a time.

**Multi-step rollout prediction**

Purpose: long-horizon forecasting. This is the main task loss and should use the same reset policy (\pi) you plan to use at inference.

### Sparsity and structure losses

**Sparse-group code penalty**
[
L_{\text{sg}}=\sum_t\left(
\lambda_1 |z_t|*1 + \lambda_2 \sum*{b=1}^B |z_{t,G_b}|_2
\right),\qquad z_t=E(x_t).
]

Purpose:

* (\ell_1) gives elementwise sparsity.
* Group-(\ell_2) encourages whole blocks/groups to turn on or off together.

This is much better than plain (\ell_1) if you want latent groups to line up with basin-like structure.

**Group persistence**
First define a soft basin/group code
[
\bar a_b(z_t)=\frac{|z_{t,G_b}|*2}{\sum_j |z*{t,G_j}|*2+\varepsilon}.
]
Then
[
L*{\text{pers}}=\sum_t |\bar a(z_{t+1})-\bar a(z_t)|_1.
]

Purpose: in your deterministic autonomous toy systems, a trajectory should not switch basins. So this is a very useful **unsupervised** prior: the dominant group should stay nearly constant along a trajectory.

**Low-entropy group activity**
[
L_{\text{ent}}=\sum_t H(\bar a(z_t)).
]

Purpose: encourage one dominant group instead of many mildly active groups. I would keep this mild, because too much entropy pressure can force artificial hard assignments.

**Off-block penalty** for soft block-sparse (K)
[
L_{\text{block}}=|(1-M)\odot K|_1
]
or (|\cdot|_F^2).

Purpose: explicitly encourage block structure in (K). This is the “Block Forcer” analogue in your one-(K) setting.

### Reset-aware consistency losses

**Projection / chart-consistency loss**
[
L_{\text{proj}}
===============

\sum_t\sum_{h\in \mathcal H_{\text{short}}}
\left|E!\left(D!\left(K^h z_t\right)\right)-K^h z_t\right|_2^2.
]

Purpose: make (E\circ D) behave like a projector onto the valid latent manifold over short rollouts. This is the most useful auxiliary loss if you want event-triggered reencoding, because it gives you a meaningful trigger score.

**Temporal consistency**
A simple tcKAE-style version is
[
L_{\text{tc}}
=============

\sum_t\sum_{0<i<j\le H}
\left|K^{j-i}E(x_{t+i})-E(x_{t+j})\right|_2^2.
]

Purpose: if the learned latent space is genuinely Koopman-like, predictions to the same future time should agree no matter which earlier point you start from. tcKAE uses exactly this kind of idea to strengthen Koopman invariance in latent space. ([ar5iv][2])

### Stability loss

For your dissipative multistable toys, I would prefer a **stability** penalty over a strict orthogonality penalty:
[
L_{\text{stab}}=\max(0,\rho(K)-r_{\max})^2.
]

Purpose: keep long rollouts from exploding.

I would only use an orthogonality/norm-preservation term if the underlying toy system has approximately conservative or oscillatory dynamics. Orthogonality has been used in KAE variants to improve long-horizon stability, but it is less natural for strongly attracting fixed-point systems. ([ar5iv][3])

### Recommended combined objectives

For **dense (K)**:
[
L_{\text{dense}}
================

L_{\text{rec}}
+\lambda_{\text{lin}}L_{\text{lin}}
+\lambda_{\text{roll}}L_{\text{roll}}^\pi
+\lambda_{\text{sg}}L_{\text{sg}}
+\lambda_{\text{proj}}L_{\text{proj}}
+\lambda_{\text{tc}}L_{\text{tc}}
+\lambda_{\text{stab}}L_{\text{stab}}.
]

For **hard block-diagonal (K)**:
[
L_{\text{hard-blk}}
===================

L_{\text{dense}}
+\lambda_{\text{pers}}L_{\text{pers}}
+\lambda_{\text{ent}}L_{\text{ent}}.
]

For **soft block-sparse (K)**:
[
L_{\text{soft-blk}}
===================

L_{\text{hard-blk}}
+\lambda_{\text{block}}L_{\text{block}}.
]

### How I would combine them in practice

I would not turn on everything at full strength from step 1.

Use a 3-stage curriculum:

1. Warm up with (L_{\text{rec}} + L_{\text{lin}} + L_{\text{sg}}).
2. Add (L_{\text{roll}}^\pi + L_{\text{proj}}).
3. Add (L_{\text{tc}}), then block/persistence terms.

That avoids the failure mode where early bad supports get frozen by structural losses.

## 3) Event-triggered re-encoding: why, and how to trigger without basin labels

The rationale is simple: no-reset rollout assumes the latent linearization stays valid globally, while reencoding only assumes it stays valid locally. Fathi et al. make exactly this local-vs-global distinction and show periodic reencoding as the middle ground between no reencoding and reencoding every step. They also explicitly note that reencoding is not teacher forcing, because it never injects ground-truth future states. ([ar5iv][1])

You do **not** need access to the true fixed point or basin label to decide when to reset. You only need internal evidence that the current latent rollout is leaving the learned manifold or current local chart.

### Trigger scores I would actually test

**Projection gap**
[
\delta_t^{\text{proj}}
======================

\frac{|E(D(\tilde z_{t+1}))-\tilde z_{t+1}|*2}
{|\tilde z*{t+1}|*2+\varepsilon},
\qquad \tilde z*{t+1}=K z_t.
]

Interpretation: if decode→reencode changes the latent state a lot, then the latent rollout has drifted off the autoencoder manifold or away from the current chart.

This is the most general trigger and the one I would start with.

**Group ambiguity**
Using the soft group code (\bar a(z_t)),
[
\delta_t^{\text{amb}} = H(\bar a(z_t))
\quad\text{or}\quad
1-\max_b \bar a_b(z_t).
]

Interpretation: if no block clearly dominates, the current code is not confidently assigned to one local regime.

This is especially useful for block or soft-block models.

**Off-block spillover** for soft block-sparse (K)
Let (g_t = \arg\max_b \bar a_b(z_t)). Then
[
\delta_t^{\text{spill}}
=======================

1-
\frac{|(Kz_t)*{G*{g_t}}|_2}{|Kz_t|_2+\varepsilon}.
]

Interpretation: if the linear step is pushing mass outside the currently dominant block, the current chart is probably becoming invalid.

**Support-margin trigger**
[
\delta_t^{\text{margin}}
========================

\frac{\min_{i\in \operatorname{supp}(z_t)} |z_{t,i}|}
{\tau_t+\varepsilon}.
]

Interpretation: if active coefficients are close to threshold, the support is fragile and a reset may cleanly reseat it.

### Reset policy

A practical policy is
[
r_t
===

\mathbf 1!\left[
\delta_t^{\text{proj}}>\tau_{\text{proj}}
;\vee;
\delta_t^{\text{amb}}>\tau_{\text{amb}}
;\vee;
\delta_t^{\text{spill}}>\tau_{\text{spill}}
\right]
]
with

* a **minimum dwell time** after a reset, to avoid chatter,
* and a **maximum interval** (m_{\max}), so you still reset eventually even if the scores stay quiet.

That hybrid policy is usually better than purely periodic or purely event-driven resets.

### How to choose thresholds without labels

Do it from the training/validation distribution of the scores:

* choose a target reset budget,
* set thresholds by quantiles of the trigger statistics,
* or tune them on forecast-vs-compute tradeoff.

So the decision rule is label-free. The only thing it uses is model self-consistency.

### What to compare experimentally

I would compare:

* no reencoding,
* fixed periodic (m\in{1,2,4,8,\infty}),
* event trigger using only (\delta^{\text{proj}}),
* event trigger using (\delta^{\text{proj}}+\delta^{\text{amb}}),
* hybrid event + max interval.

Match average reset counts when comparing methods, otherwise event-triggered resets may win just because they spend more compute.

### How to validate whether the trigger is meaningful

Because your toy systems are known, you can evaluate triggers **after training** against true geometry:

* distance to the separatrix,
* true attractor label,
* true local latent linearization error map
  [
  \ell(x)=|E(f^\Delta(x)) - K E(x)|.
  ]

So training and inference stay label-free, but evaluation can tell you whether the trigger is actually detecting “leaving a local chart.”

A good sign is: resets are sparse, concentrated near difficult transients or near the separatrix. A bad sign is: resets fire constantly inside an easy basin.

## 4) Alterations to the LISTA encoder

Your current encoder is
[
u^{(0)} = \operatorname{shrink}(c, \alpha/L),
\qquad
z = \operatorname{ReLU}!\bigl(\operatorname{shrink}(S u^{(0)} + c, \alpha/L)\bigr),
]
with (c=\text{MLP}(x)), learned (S), and (L) computed from the decoder dictionary.

I would test five changes.

### A. Add a dynamics prior to the encoder

This is the most important change.

At a reset time, you already have the predicted latent (\tilde z_t). Do not ignore it. Reencoding should solve “sparse inference near a predicted latent,” not “fresh encoding from scratch.”

A simple variant is
[
c_t = W x_t + R_\psi(x_t) + B\tilde z_t,
]
then
[
u_t^{(0)}=\operatorname{shrink}(c_t,\tau_t),
\qquad
z_t=\mathcal T(Su_t^{(0)}+c_t,\tau_t),
]
where (\mathcal T) is your chosen thresholding operator.

This makes the reset behave like a proximal correction around the current forecast, which is exactly what you want for event-triggered reencoding.

### B. Test signed vs nonnegative codes

Your final ReLU enforces (z\ge 0). That can help interpretability, but it may also be too restrictive.

I would test three variants:

1. **Current nonnegative code**
   [
   z=\operatorname{ReLU}(\operatorname{shrink}(\cdot)).
   ]
2. **Signed code**
   [
   z=\operatorname{shrink}(\cdot).
   ]
3. **Sign-split code**
   [
   v=\operatorname{shrink}(\cdot),\qquad
   z=[\operatorname{ReLU}(v),,\operatorname{ReLU}(-v)].
   ]

My bet is that **sign-split** is the best compromise. It preserves sign information while keeping support semantics clean.

### C. Use more than one LISTA step

One refinement step may be too shallow, especially near separatrices or after long latent rollouts.

I would test 1, 2, and 4 LISTA steps. I would also test a momentum version. LISTA variants with support selection, adaptive thresholds, and momentum all exist precisely because convergence speed and robustness matter; HyperLISTA argues momentum helps convergence and generalization to unseen sparsity patterns, while support selection and EBT improve convergence/adaptivity in LISTA-style models. ([arXiv][4])

### D. Make thresholds sample-dependent

A single global threshold (\alpha/L) is unlikely to be optimal everywhere in phase space.

Inspired by EBT-LISTA, make the threshold depend on the current input and possibly the dynamics prior:
[
\tau_t
======

\frac{\alpha_0
+\alpha_1|x_t-D(u_t^{(0)})|_2
+\alpha_2|u_t^{(0)}-\tilde z_t|_2}
{L}.
]

You can also do this blockwise:
[
\tau_{b,t}=\frac{\alpha_b}{L_b},
\qquad
L_b = |D_{G_b}^\top D_{G_b}|_2.
]

EBT-LISTA’s point is that thresholding should adapt to the sample and reconstruction error, not be globally shared across all inputs. ([ar5iv][5])

### E. Make the shrinkage group-aware

If you want block-structured (K), your encoder should also have block-aware sparsity.

A good operator is sparse-group shrinkage:
[
v_{G_b}\leftarrow
\left(1-\frac{\tau_g}{|v_{G_b}|*2}\right)*+ v_{G_b},
\qquad
z \leftarrow \operatorname{shrink}(v,\tau_1).
]

This gives you two levels:

* whole groups turn on/off,
* individual coordinates inside a group can still be sparse.

That is much closer to “implicit basin code + within-basin coordinates” than pure elementwise shrinkage.

### F. Add support selection, but only as a later-stage experiment

LISTA support selection keeps the largest-magnitude entries out of thresholding and has theoretical/practical convergence benefits. I would test it, but not as the very first variant, because near separatrices it can prematurely lock onto the wrong support. ([arXiv][4])

A structured version is better:

* first select top-1 or top-2 groups by (|v_{G_b}|_2),
* then threshold within those groups.

That is more aligned with your basin objective than plain top-(k) entry selection.

### G. Keep the pre-code MLP on a short leash

For identifiability experiments, I would explicitly compare:

* **free MLP pre-code**,
* **dictionary-tied pre-code** initialized near (D^\top x/L),
* **hybrid pre-code**
  [
  c = W x + R_\psi(x)
  ]
  with small (R_\psi).

If the MLP is too expressive, it can bypass the sparse-coding bias and you stop learning something interpretable.

## 5) Other things I think are important

### Latent dimension is a real axis, not a nuisance hyperparameter

Even though the physical state is 2D, I would still sweep (p), for example (p\in{8,16,32,64}). Recent KAE work explicitly argues that too-small latent spaces can be limiting and that latent dimension inflation can improve expressive power and long-term forecasting. Also, in multistable settings you should not expect the useful Koopman measurement space to be as small as the state dimension. ([ar5iv][3])

### Add a decoder-coherence penalty

Because your decoder is a normalized linear dictionary, sparse-code identifiability depends heavily on atom coherence. I would test
[
L_{\text{coh}}=
\sum_{i\neq j} (d_i^\top d_j)^2
]
or
[
L_{\text{coh}}=|D^\top D-I|_{F,\text{off}}^2.
]

Without this, the encoder may use many equivalent sparse supports.

### Use true basin labels only for evaluation

Since the systems are toy and known, compute:

* true attractor reached by each initial condition,
* distance to separatrix,
* local Jacobian field if useful.

Then evaluate:

* final-attractor prediction accuracy,
* NMI/ARI between dominant latent group and true basin label,
* within-trajectory group stability,
* trigger score vs separatrix distance,
* forecast error stratified by distance to separatrix.

That gives you a clean identifiability study even if training is fully unsupervised.

### Oversample hard initial conditions

Most models will look good deep inside a basin. The real differences show up:

* near the separatrix,
* near saddles,
* in long transients before settling.

So I would intentionally oversample those regions.

### If some toy systems are symmetric, exploit that

Pan and Duraisamy specifically discuss symmetry among invariant sets. If you build symmetric toys, it is worth testing whether corresponding blocks can be tied by permutation or sign symmetry, or whether an untied model rediscovers equivalent blocks on its own. That is a very strong identifiability probe. ([arXiv][6])

## My recommended experiment order

I would run the study in this order:

1. Dense (K), current encoder, no resets.
2. Dense (K), fixed periodic resets.
3. Dense (K), projection-gap event trigger.
4. Dense (K), dynamics-aware LISTA and sample-dependent thresholds.
5. Hard block-diagonal (K) with sparse-group code and group persistence.
6. Soft block-sparse (K) with off-block penalty.
7. Repeat 5–6 with sign-split codes and 2–4 LISTA steps.
8. Sweep latent dimension (p) and block count (B).

If I had to guess the outcome on your 2D multistable toys:

* **Forecasting only:** dense (K) with resets may already be very strong.
* **Forecasting + interpretable basin structure:** soft block-sparse (K) is the most likely winner.
* **Pure identifiability:** hard block-diagonal (K) with group-aware sparsity and persistence is the cleanest test.

The single change I would prioritize first is **making reencoding dynamics-aware**, i.e. feeding the predicted latent (\tilde z_t) into the LISTA inference step. That is the most direct way to turn “periodic reencoding” from a heuristic into a principled sparse correction step.

[1]: https://ar5iv.org/pdf/2310.15386 "https://ar5iv.org/pdf/2310.15386"
[2]: https://ar5iv.labs.arxiv.org/html/2403.12335v3 "https://ar5iv.labs.arxiv.org/html/2403.12335v3"
[3]: https://ar5iv.org/pdf/2503.12930 "https://ar5iv.org/pdf/2503.12930"
[4]: https://arxiv.org/pdf/1808.10038 "https://arxiv.org/pdf/1808.10038"
[5]: https://ar5iv.org/html/2112.10985v2 "https://ar5iv.org/html/2112.10985v2"
[6]: https://arxiv.org/html/2304.11860v4 "https://arxiv.org/html/2304.11860v4"
