Status note as of `2026-04-09`:

- The reducer-side evaluation tooling for support-freeze rollouts, switch-
  timing summaries, effective-Jacobian/operator-family summaries, and the
  visual diagnostic suite is implemented locally.
- On the fixed `17`-system live branch, use `1` seed by default for
  interpretability diagnostics while ranking methods and metrics. Promote a
  shortlisted effect to multi-seed confirmation only after the seed-`0`
  result looks strong enough to matter for the paper.
- The remaining gap in this study plan is now primarily experimental rather
  than infrastructural: rerun those diagnostics on the fixed shortlist roots
  and decide whether the honest paper claim lands at exact-support reuse, at
  support-family / dominant-group alignment, or at a weaker but still useful
  symmetry-aware operator-family alignment.

I would make the study answer **three separate questions**, not one:

1. **Does support identify basin?**
2. **Does support identify a distinct local linear law?**
3. **Do support switches track control-induced basin switches?**

That separation matters because with one global (K), a model can have support patterns that correlate with basins **without** actually using those supports as distinct linearizations. Also, “one exact support per basin” is a very strong target: sparse dictionaries are only identifiable under restrictive conditions and only up to sign/permutation, while multistable Koopman settings with multiple disjoint invariant sets often require stitched or discontinuous observables in finite-dimensional constructions. So I would treat **exact support uniqueness** as the strongest possible outcome, but I would also explicitly measure **support families** and **operator families** as weaker, more realistic outcomes. 

## 1) Define the right ground truth

For each toy system, I would label every state (x) by its **basin ID**
[
b(x)\in{1,\dots,B},
]
where (b(x)) is the attractor reached by the **unforced** system started from (x). For controlled transfer trajectories, I would still label each intermediate state by “which attractor would it reach if the control were turned off now?” That gives you a consistent notion of the basin that the encoder is supposed to detect.

Then I would build three evaluation splits:

* **deep-basin states**: far from separatrices,
* **boundary states**: near separatrices/saddles,
* **controlled transfer trajectories**: states driven across separatrices.

For interpretability, this split is crucial. A model that really learns basin-specific supports should be most stable deep inside a basin, most ambiguous near boundaries, and should switch supports cleanly during controlled basin transfers. I would also include both **symmetric** and **symmetry-broken** toy systems; in symmetric systems, two basins may legitimately learn permuted-equivalent support templates rather than unrelated ones, and Pan–Duraisamy explicitly show symmetry can change what a good Koopman construction looks like and how efficiently it can be learned. ([arXiv][1])

## 2) Decide what “support” means before you measure it

Because LISTA outputs continuous coefficients, you need a support definition:
[
s_\tau(x)_i=\mathbf 1{z_i(x)>\tau},
\qquad z(x)=E(x).
]

I would never report results at only one threshold. I would sweep (\tau) over a small range, for example:

* absolute threshold,
* relative threshold (\tau=\gamma\max_i z_i),
* top-(k) active set.

That tells you whether “unique supports” are real or are artifacts of a threshold choice.

I would also define a **group-level support** if you are using hard or soft block structure:
[
g(x)=\arg\max_{m} |z_{G_m}(x)|_2.
]
This gives you a more stable fallback notion: even if exact supports differ by one or two atoms, a basin might still map to one dominant group. That is often the more meaningful interpretability object. If you want that behavior during training, sparse-group penalties are the standard way to encourage both group activation and within-group sparsity. 

## 3) Train a model grid that lets you separate “emergent” from “forced”

Keep the architecture fixed except for the axes that matter to the hypothesis:

* (K): dense, hard block-diagonal, soft block-sparse.
* reset policy: none, periodic, event-triggered.
* encoder: current 1-step LISTA, plus one or two sharper variants such as support-selection LISTA.

LISTA itself is a learned approximation to sparse coding/ISTA, and Chen et al.’s LISTA-SS variant explicitly biases the inference toward crisp large-magnitude supports. That makes support-selection variants useful **ablations**: if unique basin supports only appear after aggressive support selection, then the support interpretability is being manufactured rather than naturally discovered. Periodic reencoding is also worth treating as a core axis, since Fathi et al. introduced it precisely as an inference-time correction for long-horizon latent drift. 

For the live fixed-`17` diagnostic branch, I would not train every setting
with multiple random seeds up front. I would use one seed first to rank
candidate axes cheaply, then rerun only the strongest shortlisted effects with
multiple seeds. Because sparse dictionaries are only identifiable up to
sign/permutation, raw support IDs are not comparable across runs unless you
align atoms or groups post hoc. So once a setting is promoted to multi-seed
confirmation, align models across seeds with a Hungarian matching based on
decoder-atom correlations and basin-conditioned activation maps. Without this
step, apparent instability across runs may just be the usual sign/permutation
ambiguity rather than a real interpretability failure.

## 4) Diagnose whether supports identify basins

This is the first layer.

Let (S) be the random support pattern and (B) the true basin label.

### Exact-support metrics

For each basin (b), estimate the empirical distribution (p(s\mid b)). Then report:

[
U_{\text{exact}} = \frac{1}{B}\sum_{b=1}^B \max_s p(s\mid b),
]
the average mass of the most common support in each basin.

Also report:
[
H(S\mid B),\qquad H(B\mid S),\qquad I(S;B),
]
or normalized mutual information.

Interpretation:

* low (H(S\mid B)): each basin uses few supports,
* low (H(B\mid S)): each support points to one basin,
* high (U_{\text{exact}}): close to one exact support per basin.

This is the cleanest quantitative test of your ideal outcome.

### Support-family metrics

Exact support may be too brittle. So cluster supports by Jaccard distance, Hamming distance, or dominant group (g(x)), then repeat the same metrics with a support-family label (C) instead of exact support (S).

For symmetric or near-symmetric systems, I would also repeat that analysis after aligning decoder atoms across basins or seeds. If two basins use supports that are the same up to a permutation, sign flip, or other simple symmetry-aligned relabeling, that should count as one aligned support family rather than as a false failure of reuse.

This lets you distinguish:

* **strong result**: one exact support per basin,
* **good result**: one support family / dominant group per basin,
* **weak result**: many unrelated supports per basin.

### State-space stratification

Compute all of those metrics as functions of:

* distance to separatrix,
* distance to the attracting fixed point,
* time-to-attractor.

That matters because the model may only learn unique supports **near the attractor**, not over the whole basin. If (H(S\mid B)) is low near fixed points but high in transients, then the model learned local linearization around attractors, not basin-wide support uniqueness.

## 5) Diagnose whether supports identify distinct local linearizations

This is the most important part.

With one global (K), the support only matters interpretably if it selects a distinct local linear law. I would test that in **several complementary ways**.

One important caution: do **not** make “different basins must have clearly different eigenvalues” part of the success criterion. Two basins can share very similar local spectra while differing mainly in orientation, symmetry transform, or the chart used by the encoder to represent the same intrinsic local law. Pan and Duraisamy make the symmetry point in a Koopman setting for multiple disjoint invariant sets, and Salova et al. show that symmetry can organize Koopman spectra and block structure without forcing each invariant set to look unrelated in raw coordinates. ([arXiv][1], [arXiv][3])

### A. Support-conditioned operators in latent space

For each support class (c) (exact support, support family, or dominant group), fit a post-hoc local operator:
[
A_c
===

\arg\min_A
\sum_{t:,c_t=c}
|z_{t+1}-Az_t|_2^2.
]

Now compare three things:

1. the global one-(K) predictor,
2. support-conditioned local fits (A_c),
3. true-basin local fits (A_b), where
   [
   A_b
   =
   \arg\min_A
   \sum_{t:,b_t=b}
   |z_{t+1}-Az_t|_2^2.
   ]

If the support story is real, then support-conditioned operators should nearly match basin-conditioned operators in both forecast error and geometry.

I would report:

[
\Delta_{\text{within}}=
\mathbb E[|A_c-A_{c'}|*F \mid b(c)=b(c')],
]
[
\Delta*{\text{between}}=
\mathbb E[|A_c-A_{c'}|_F \mid b(c)\neq b(c')].
]

You want (\Delta_{\text{between}}\gg \Delta_{\text{within}}).

Before interpreting those raw distances, I would separate three cases explicitly:
[
A_{b_1}\approx A_{b_2},
\qquad
A_{b_2}\approx Q^{-1}A_{b_1}Q,
\qquad
A_{b_1}\not\sim A_{b_2}.
]

These correspond to:

* essentially the same local law,
* the same local law up to a change of basis or symmetry transform,
* genuinely different local laws.

That distinction matters because two basins can have nearly identical attraction rates but rotated eigendirections. In that case their fitted local operators may be **similar matrices** rather than obviously different ones.

### B. Operator equivalence up to similarity

So I would not compare basin-conditioned or support-conditioned operators only with
[
|A_b-A_{b'}|_F.
]

I would also compare them after the best restricted alignment:
[
d_{\mathrm{sim}}(A_b,A_{b'})
=
\min_{Q\in \mathcal Q}
|A_{b'}-Q^{-1}A_bQ|_F,
]
where (\mathcal Q) is a restricted class of transforms.

For these toy systems, I would try:

* orthogonal (Q) first, to capture rotated or reflected eigendirections,
* signed permutation (Q), to capture latent-axis relabeling,
* optionally general invertible (Q), but I would treat that as a weaker and less interpretable equivalence notion.

If raw distance is large but aligned distance is small, then the two basins do **not** have different intrinsic local dynamics; they have similar dynamics written in different coordinates. I would repeat the same comparison for support-conditioned operators (A_c), not just basin-conditioned ones.

### C. Does the learned (K) itself respect the supports?

For each support (s), define the masked operator
[
K^{(s)} = P_s K P_s,
]
where (P_s) projects onto the active coordinates.

Then compare (K^{(s)}) to the post-hoc fitted (A_s). If these are close, then the one global (K) is genuinely being used as an implicit collection of support-conditioned local linearizations. If they are not close, then support may correlate with basin, but the actual learned linearization is not really “selected by support.”

This is the most direct answer to your interpretability goal.

### D. Effective state-space Jacobians

A very strong additional test is to differentiate the learned one-step predictor in state space:
[
\hat f(x)=D(KE(x))
]
or, if you reset every step for analysis,
[
\hat f_{\text{reset}}(x)=D(E(D(KE(x)))).
]

Then compute
[
J_{\hat f}(x)=\frac{\partial \hat f}{\partial x}(x).
]

If support really selects a local linearization, then states with the same support should have similar (J_{\hat f}(x)), and Jacobians should differ much more across basins than within a basin. Near each fixed point, you can even compare (J_{\hat f}(x^\star_b)) to the true Jacobian of the toy system there. This gives you an interpretable state-space notion of “learned local linearization,” not just a latent one.

I would also report **spectral similarity** separately from **directional similarity**. For each learned local operator or effective Jacobian, record:

* eigenvalues,
* leading eigenvector directions,
* and principal angles between invariant subspaces when the spectrum is repeated or nearly repeated.

In `2D`, a simple directional metric is the angle between normalized leading eigenvectors:
[
\theta(b,b')=\arccos\!\bigl(|v_b^\top v_{b'}|\bigr).
]
Use the absolute value because eigenvectors are sign-ambiguous. This lets you call a pair “same contraction rates, different orientation” rather than incorrectly collapsing it into “same local linearization.”

## 6) Diagnose support stability and switching on trajectories

Now test the temporal story directly.

### Within-basin stability

For uncontrolled trajectories that stay in one basin, measure:

[
\text{Persistence}=\Pr(s_{t+1}=s_t \mid b_{t+1}=b_t),
]
and the average Jaccard similarity
[
J(s_t,s_{t+1})=\frac{|s_t\cap s_{t+1}|}{|s_t\cup s_{t+1}|}.
]

High forecasting accuracy with low support persistence means the code is not an interpretable basin indicator; it is just using support opportunistically.

### Controlled basin transfers

For a controlled transfer trajectory, define the true basin-switch time
[
t^\star = \min{t: b(x_t)\neq b(x_{t-1})},
]
where (b(x_t)) is the basin reached if control is removed at time (t).

Then define the inferred support-switch time (\hat t) from either exact support or dominant group changes. Report:

* detection delay (\hat t-t^\star),
* false switch rate before (t^\star),
* chatter rate after (\hat t),
* dwell time before and after the switch.

Interpretation is simple:

* good model: one stable support before crossing, one switch near (t^\star), one stable new support after,
* bad model: support chatter everywhere, or switches that do not line up with basin transfer.

Periodic or event-triggered reencoding is especially relevant here, because it is the mechanism that gives the encoder chances to reselect the current local chart rather than staying locked to a drifting latent rollout. ([arXiv][2])

## 7) Run counterfactual support interventions

These are the most interpretable tests.

For each basin (b), define a canonical support or group template (s_b^\star), for example the most common support deep inside that basin.

For symmetry-related basins, I would define that template in two ways:

* a raw canonical support template,
* and an alignment-aware template after Hungarian matching on decoder atoms.

That lets you distinguish unrelated supports from supports that are the same up to permutation or sign.

Then for a state (x) in basin (b), build:

[
z = E(x),\qquad
z^{\text{own}} = P_{s_b^\star} z,\qquad
z^{\text{wrong}} = P_{s_{b'}^\star} z ;; (b'\neq b).
]

Now compare forecast quality under:

* no projection,
* own-basin support projection,
* wrong-basin support projection.

If support really selects a basin-local linearization, then:

* projecting to the correct basin support should preserve or improve prediction,
* projecting to the wrong basin support should sharply hurt prediction, and often push the forecast toward the wrong attractor.

A second intervention is to **freeze the support** along rollout. Deep inside a basin, a true basin-local support should tolerate this well for a while. If freezing support immediately hurts even deep inside the basin, then the support is probably not functioning as a stable basin-local chart.

## 8) Visual diagnostics

I would make the paper/report visually centered around six plots:

1. **Phase portrait colored by true basin**.
2. **Same phase portrait colored by dominant support / dominant group**.
3. **Support entropy map** over the phase plane.
4. **Support-switch raster** for controlled transfer trajectories.
5. **Basin (\leftrightarrow) support confusion matrix / Sankey diagram**.
6. **Operator-distance heatmap** for (A_c) or (J_{\hat f}(x)).

The most revealing plot is usually the phase portrait colored by support with the separatrix overlaid. If the support partition is meaningful, you should see large coherent regions that line up with basins, plus ambiguity only near the boundary.

## 9) How to interpret outcomes

I would use this rubric.

### Strong success

Deep inside each basin, one exact support dominates; (H(B\mid S)) and (H(S\mid B)) are both low; support-conditioned operators are tight within basin and distinct across basins even after checking for simple similarity transforms; support switches align with control-induced basin changes.

### Partial success

Each basin has a unique dominant group, a small support family, or a symmetry-aligned support class, but not one exact support. Supports within the same family induce nearly the same local operator, or operators that become nearly the same after a simple orthogonal or signed-permutation alignment. This still supports the “support selects local linearization” story, just at the family/group or aligned-equivalence level rather than the exact-support level.

### Failure

The same supports appear in multiple basins, supports change frequently within a single basin, or support-conditioned operators do not separate by basin even after symmetry-aware / basis-aware alignment analysis. In that case the model may still forecast well, but it is not learning the interpretable basin-specific support mechanism you want.

## 10) The single most important methodological choice

I would not make **exact support uniqueness** the sole criterion.

I would make the primary claim:

[
\text{basin}
\;\longrightarrow\;
\text{support family / dominant group / aligned support class}
\;\longrightarrow\;
\text{distinct local operator family}.
]

That is the right target because:

* exact sparse supports are basis-dependent,
* dictionaries have sign/permutation ambiguity,
* symmetry-related basins may share intrinsic local laws up to a simple transform,
* and multistable Koopman structure may naturally be stitched from local pieces rather than represented by one perfectly rigid support per basin. 

If I had to pick only **three** diagnostics, I would pick:

1. (H(B\mid S)) and (H(S\mid B)) deep inside basins,
2. support-switch alignment on controlled transfers,
3. support-conditioned operator separation (A_c) vs basin-conditioned (A_b), using both raw distance and similarity-aligned distance.

Those three together tell you whether the encoder is learning **supports that are basin-specific, temporally meaningful, and actually tied to distinct learned linearizations**.

[1]: https://arxiv.org/pdf/2304.11860 "On the lifting and reconstruction of nonlinear systems with multiple invariant sets"
[2]: https://arxiv.org/abs/2310.15386?utm_source=chatgpt.com "Course Correcting Koopman Representations"
[3]: https://arxiv.org/abs/1904.11472 "Koopman Operator and its Approximations for Systems with Symmetries"
