# Support Object Glossary

Date: April 26, 2026

## Purpose

This note gives a paper-facing glossary for the support objects used in the
fixed-`17` benchmark analysis. The goal is to separate:

- how a continuous latent state is converted into a binary support mask,
- which higher-level support objects are built from that mask,
- and which claims are currently supported by the saved experiments.

The most important conceptual distinction is between a **support as a region
label** and a **support as a dynamical selector**. A support is a good region
label when observing it tells us which basin, or which coherent part of a
basin, contains the current state. That is a static membership claim. A support
is a useful dynamical selector only if the same active set also identifies
which latent coordinates, or which centered local linear law, should be used
for prediction. The first property does not imply the second: an encoder could
use some active coordinates as basin markers while the next-step dynamics are
carried by other coordinates, by continuous coefficient values within the same
support, or by interactions across several columns of the learned Koopman
transition. This is why the paper needs both support-label metrics and
support-routed forecasting diagnostics.

The operational distinction is:

- `absolute:0.001`, `relative:0.1`, and `topk:8` are **support definitions**
  that turn a latent vector into a binary mask;
- exact support, support family, and dominant group are **support objects**
  built from those masks or from grouped latent structure.

## Support Definitions

| Term | Meaning | Main advantage | Current strongest evidence |
| --- | --- | --- | --- |
| `absolute:0.001` / `S_abs` | Mark a latent coordinate active when its magnitude is above `0.001`. | Direct and easy to interpret; best for asking whether one basin is represented by one exact support away from basin boundaries. The active count `|S_abs|` is threshold-dependent and can vary by state, basin, system, seed, and latent scale. | On the selected states far from basin boundaries in the locked final packet, the promoted dense LISTA finalist reaches `H(B\|S_abs)=0.0000`, `H(S_abs\|B)=0.0543`, `U_exact=0.9923`, `H(B\|F_abs)=0.0000`, versus the matched sparse MLP at `H(S_abs\|B)=0.2449` and `U_exact=0.9772`. This is the cleanest exact-support identifiability result. |
| `relative:0.1` | Mark a latent coordinate active when its magnitude is at least `10%` of the largest latent magnitude for that state. | Scale-relative; still meaningful on dense non-sparse controls where exact zeros are not expected. | Very useful for centered local-law diagnosis, but too fragmented as a deployment router. In the centered local-law packet it gave strong wins away from basin boundaries, but in the self-routed packet the exact-support router was skipped `160/160` times per root on the deepest evaluation slice. |
| `topk:8` | Mark exactly the `8` largest-magnitude latent coordinates active. | Fixed support size; much more stable as a routing signal than thresholded exact support. | This is the strongest non-oracle forecasting result. In the self-routed packet, dense LISTA exact-support `topk:8` achieves all-slice `H1000/global` median / win rate `0.228 / 0.920` for support-gated prediction and `0.275 / 0.947` for centered local-law prediction. The dense zero-sparsity MLP is much weaker at `0.924 / 0.539` and `1.000 / 0.496`. |

## Support Objects

A support object is the discrete label extracted from the encoder's latent
state after choosing a support definition. For a state `x`, the encoder
produces a continuous latent vector `z = E(x)`. A support definition such as
`absolute:0.001`, `relative:0.1`, or `topk:8` turns `z` into an active-coordinate
mask. The support object is then the object used for analysis or routing:
the exact mask itself, a support family formed by merging similar masks, or
the dominant latent group. In the metrics below, `S` denotes whichever support
object is being evaluated in that row.

| Term | Meaning | Why it matters | Current reading |
| --- | --- | --- | --- |
| Exact support | The full binary active-set mask itself. Two states match only if they activate exactly the same latent coordinates. | This is the strongest and most literal hypothesis: one basin, one support. | Exact support is the strongest object when it works, but it is brittle. It is highly positive on the `absolute:0.001` slice away from basin boundaries and is also the main object behind the successful dense LISTA `topk:8` self-routed forecasting result. It is not robust under `relative:0.1` for deployment. |
| Support family | A cluster of similar exact supports, formed by greedy Jaccard merging with threshold `0.5`. | This is the realistic fallback when exact supports differ by a few coordinates but still reflect the same underlying basin-specific pattern. | Family-level agreement with basin labels is very strong. On the selected `absolute:0.001` / states-far-from-boundaries slice, `H(B\|F_abs)=0.0000` for all four main roots in the locked final comparison. In self-routed forecasting, family routing often has extremely strong LISTA medians, but a minority of catastrophic rollouts makes it unstable as the main deployment claim. |
| Dominant group | For grouped or block-structured latents, pick the group with the largest latent-group norm. | This is the coarsest support object and a useful fallback when exact support is too fragile. It asks “which latent group dominates?” rather than “which exact coordinates are on?” | Dominant group is currently supporting evidence rather than the headline object. On the locked final packet `absolute:0.001` / states-far-from-boundaries slice, group-level agreement is decent but clearly weaker than the best exact-support read: mean `H(B\|G)` / `H(G\|B)` / `NMI` are about `0.281 / 0.072 / 0.558` for the blockdiag LISTA finalist and `0.260 / 0.062 / 0.578` for the dense soft-block LISTA finalist. |

## How To Read The Main Metrics

| Metric | Meaning | Better direction |
| --- | --- | --- |
| `H(B\|S)` | How mixed basins are inside one support object. | Lower |
| `H(S\|B)` | How fragmented a basin is across many support objects. | Lower |
| `U_exact` | Mean dominant exact-support mass per basin. | Higher |
| `H(F\|B)` | How fragmented a basin is across support families. | Lower |
| `NMI` | Mutual dependence between basin labels and the support object. | Higher |

Plain-language reading:

- `H(B|S)` asks: after I know the support object, how uncertain am I about the
  basin? Low values mean that one support object rarely mixes states from
  different basins.
- `H(S|B)` asks: after I know the basin, how uncertain am I about the support
  object? Low values mean a basin is represented by one or a few recurring
  support objects. High values mean points from the same basin are scattered
  across many different support objects.
- Example: if every point in basin A has exact support `{2, 5, 9}`, then basin
  A is not fragmented and its contribution to `H(S|B)` is near zero. If basin A
  is split across `{2, 5, 9}`, `{2, 5, 11}`, `{1, 7, 12}`, and dozens of other
  masks with comparable frequency, then knowing "this point is in basin A"
  still does not tell us which support it uses; the basin is fragmented and
  `H(S|B)` is high.

The cleanest qualitative read for static label agreement is:

- low `H(B\|S)` means one support object rarely mixes basin labels,
- low `H(S\|B)` means each basin uses only a few such objects,
- high `U_exact` means one exact support dominates within each basin,
- low `H(F\|B)` means support families agree very cleanly with basin labels.

These quantities do not test whether the support selects a useful local
predictor. They answer the question "does the support say where the state is?"
The routing diagnostics answer the separate question "does the support help
advance the state?" In the support-gated diagnostic, the support masks the
centered latent vector before applying the learned global transition, so it
tests whether the active set selects useful columns or subspaces of that
transition. In the centered local-law diagnostic, the support chooses a
post-training local linear fit, so it tests whether the same support is a good
key for a local predictor.

## Safest Current Claims

1. **Exact-support identifiability claim**

   Safe only in a qualified form: away from basin boundaries, the promoted
   dense LISTA finalist shows near-unique exact supports on the
   `absolute:0.001` slice.

2. **Support-family agreement claim**

   This is the safest broad interpretability claim. Family entropy is already
   essentially saturated on the selected states-far-from-boundaries slice, and
   family-level agreement with basin labels is much more robust than
   exact-support uniqueness.

3. **Deployment-facing forecasting claim**

   The strongest current deployment object is not thresholded exact support. It
   is dense LISTA exact-support routing under `topk:8`, which improves
   long-horizon forecasting without oracle basin labels.

4. **Dominant-group claim**

   Dominant group should be written as a coarse supporting view, not as the
   main paper object. It is more stable than exact support, but currently less
   discriminative than the best exact-support or support-family evidence.

## What Not To Collapse Together

- Do not treat `absolute:0.001`, `relative:0.1`, and `topk:8` as interchangeable.
- Do not treat exact support and support family as the same claim.
- Do not treat family-level alignment as proof of exact-support uniqueness.
- Do not treat the dominant group fallback as equivalent to exact-support
  routing.
- Do not treat support agreement with basin labels as proof that the support
  selects useful columns of the learned transition or a useful centered local
  law. That requires the routing diagnostics.

## Recommended Paper Usage

- Use `absolute:0.001` when making the exact-support agreement read on states
  far from basin boundaries.
- Use `topk:8` for the main non-oracle forecasting table and figure.
- Use support families for the broader support-label agreement story and for
  robustness/interpretability discussion.
- Use dominant group as a coarse fallback view, mainly in supporting text or
  appendix-level diagnostics.
