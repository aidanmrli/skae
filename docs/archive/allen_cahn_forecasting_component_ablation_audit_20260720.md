# Allen--Cahn forecasting component-ablation audit (2026-07-20)

## Concrete result

The positive full-horizon Allen--Cahn comparison estimates the effect of a
**joint sparse recipe**, not the effect of soft thresholding or an elementwise
L1 penalty separately.  The selected signed sparse arm differs from the exact
dense arm in exactly two substantive operations: `softshrink(code, 0.15)` and
an elementwise mean-absolute-latent penalty of weight `0.01` during joint
training.  The ten-seed capacity audit explicitly reaches the same boundary:
the result warrants a claim about the joint sparse treatment, but not about
either component alone.

This is nevertheless a tight method comparison.  Dense and signed arms have
the same allocated parameter count and tensor shapes, the same effective
forward parameter count, tanh convolutional trunks, linear pre-code, decoder,
full trainable 2048-by-2048 Koopman matrix, identity initialization, optimizer,
zero weight decay, data, horizon, and update budget.  With one LISTA loop, the
learned LISTA recurrence matrix receives only the zero initialization as its
input and is inert; the signed encoder is therefore exactly a one-pass soft
threshold rather than a deeper iterative architecture.

The declared four-recipe development screen supplies useful robustness, but
not component attribution.  The recipe names `l1` and `l3` mean latent
consistency weights `0.1` and `0.3`; they do **not** denote L1 sparsity.  The
elementwise sparse loss remains `0.01` for every signed recipe.  Descriptively,
the signed arm has a lower mean than the dense arm in all 16 same-recipe cells
(four recipes times H160/H200 times through-horizon mean/terminal MSE), with
at least 2/3 paired wins in every cell.  Across the four opened development
recipes, signed-versus-dense reductions ranged from 4.4682% to 29.8276% for
the through-horizon mean (2/3 to 3/3 paired wins) and from 4.7582% to 10.8098%
for the terminal state (2/3 to 3/3 paired wins).  These ranges are descriptive
development-screen robustness, not confirmatory or component-causal evidence.
The selector independently chose `uniform_l1` for both arms.  Its report-half
results were:

| Cell | Dense | Signed sparse | Reduction | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| H160 through-horizon mean | 0.0470420 | 0.0449401 | 4.47% | 3/3 |
| H160 terminal | 0.0608244 | 0.0577865 | 4.99% | 2/3 |
| H200 through-horizon mean | 0.0510701 | 0.0486380 | 4.76% | 2/3 |
| H200 terminal | 0.0731957 | 0.0686168 | 6.26% | 2/3 |

The fresh ten-seed confirmation retained lower means in all four cells.  Its
reductions in the arithmetic mean over forecast states through the stated
horizon were 6.30% at H160 and 5.48% at H200, both with 8/10 wins and marginal
seed intervals above zero.  Terminal reductions were only 2.38% and 3.22%,
both 7/10, with intervals crossing zero.  The frozen four-cell conjunction
therefore failed and the conditional holdout stayed unopened.  Nothing in the
confirmation separates thresholding from the L1 term.

## Exact factors in the full-H200 screen

Factors held fixed in all 36 development runs were `d_x=512`, `d_z=2048`,
tanh convolutional activations, circular padding, two convolutional blocks,
one LISTA loop, a full dense trainable Koopman matrix, Koopman learning rate
`1e-6`, identity Koopman initialization, Adam, zero weight decay, zero Koopman
stability penalty, 2,000 autoencoder-pretraining updates, 3,500 joint updates,
batch 8, H200 training windows, shared data, and autonomous rollout without
periodic reencoding.

The factors that actually varied were:

| Axis | Dense | Signed | Sign-split |
| --- | --- | --- | --- |
| encoder output | linear code | soft threshold at 0.15 | threshold 1024 coefficients at 0.0001, then positive/negative split to 2048 |
| elementwise L1 weight | 0 | 0.01 | 0 |
| temporal group penalty | 0 | 0 | 0.001 |
| intrinsic pre-code width | 2048 | 2048 | 1024 |
| realized active density, selected screen | 0.9973 | 0.5699 | 0.4990 after sign split; pre-split groups about 0.998 active |

The sign-split arm is not a clean third point on a scalar sparsity axis.  It
changes intrinsic width and penalty type, and its approximately 50% output
density is largely structural mutual exclusion between positive and negative
copies rather than genuine sparse pre-split coefficients.  It cannot identify
which component makes the signed arm work.

The exact recipe grid was `uniform_l1` = uniform forecast weighting and latent
consistency weight 0.1, `uniform_l3` = uniform and 0.3, `late_l1` =
`late_balanced` and 0.1, and `late_l3` = `late_balanced` and 0.3.
`late_balanced` uses one half uniform trajectory loss, one quarter H160 loss,
and one quarter terminal loss.  These are optimization-objective ablations,
not sparsity-component ablations.  Uniform weighting and latent weight `0.1`
were selected for all three arms under the frozen one-percent tie rule.

One subtle stage difference also matters.  Soft thresholding is in the signed
forward path during both pretraining and joint training.  The elementwise L1
penalty is applied only by the joint-training loss; the autoencoder pretraining
loss does not receive that term.  In contrast, the sign-split temporal-group
penalty is applied in both stages.

## Existing partial ablations and why they do not identify the current gain

1. **Old threshold/depth sweep.**  The directory
   `/network/scratch/l/lia/skae/allen_cahn_lista_stiffness_50k_20260629/`
   contains 54 completed threshold-only LISTA runs: three seeds, loop counts
   1/2/4, thresholds 0, 0.0001, 0.0003, 0.001, 0.003, and 0.01, and zero
   explicit sparsity loss.  For the one-loop rows, H200 through-horizon mean
   MSE was 1.0753 at threshold zero, 1.0633 at 0.0001, and 1.0847 at 0.01;
   the corresponding terminal MSEs were 1.2421, 1.1880, and 1.2877.
   Threshold 0.01 produced about 49% near-zero coordinates in the final
   training-batch diagnostic but did not improve either H200 metric over
   threshold zero.
   This is a real threshold-only sensitivity, but it used the obsolete
   physical-time-4 regime, only 48 training trajectories per data seed,
   different Koopman initialization, no prospective selection/confirmation,
   and models that were dramatically worse than persistence.  It is not
   credible attribution for the current physical-time-20 gain.

2. **Focused one-seed architecture screen.**  The six signed-LISTA candidates
   under
   `/network/scratch/l/lia/skae/allen_cahn_rebuttal_20260719/focused_tuning/`
   crossed thresholds 0.03/0.10/0.15 with elementwise penalties 0.003/0.01,
   but used two LISTA loops, one tuning seed, H40 training, and no zero-penalty
   row.  The effect of increasing the penalty was nonmonotone across
   thresholds, and none of these signed-LISTA candidates passed the family
   noncollapse screen.  Threshold-only sign-split candidates also varied
   threshold from 0.0001 to 0.15, but they are the intrinsically 1024-D
   positive/negative-split architecture, not the current signed model.  This
   packet is exploratory hyperparameter evidence, not a clean causal ablation.

3. **Temporal-group loss screen.**  The later validation-only screen cleanly
   varied temporal-group weight 0/0.0001/0.0003/0.0006 for the sign-split
   model.  All candidates retained good support alignment and all were worse
   than dense at H200.  This establishes that the temporal penalty was not the
   sole cause of that older sign-split forecast deficit.  The current winning
   signed arm has no temporal-group loss, so this does not separate its
   threshold and elementwise L1 components.

4. **Routing and local-K screens.**  The low-rank routed, centered-affine, and
   exact half-global/half-full-local experiments vary the transition model
   after learning a sign-split representation.  Their negative results are
   valuable controls for support routing, but they do not ablate how the
   current signed representation was induced.

## Interpretation and rebuttal boundary

The defensible rebuttal statement is: under matched capacity, data, compute,
and a fail-closed dense tanh control, the **joint one-pass soft-thresholding
plus elementwise-L1 recipe** lowers the arithmetic mean rollout MSE over
forecast states through physical time 20.  The development direction holds
across the entire declared loss-weighting grid and the fresh ten-seed
through-horizon mean effect replicates, while terminal superiority and the
preregistered four-cell gate do not.

It is not defensible to say that soft thresholding alone, the L1 term alone,
basin-support alignment, LISTA depth, or invariant support subspaces caused
the forecast effect.  Calling the selected model simply a sparse treatment is
reasonable; presenting the comparison as a component-level mechanism result
is not.

## Project implications and minimal follow-up

No new run is necessary if the rebuttal keeps the joint-treatment wording and
the failed confirmation gate visible.  If a causal claim about sparsity itself
is important, the minimal prospective follow-up is a matched 2-by-2 factorial
on a new condition or dataset: threshold 0/0.15 crossed with elementwise L1
weight 0/0.01, with the same one-loop encoder, H200 objective, checkpoints,
data, seeds, and dense audit.  A zero-threshold one-loop LISTA encoder is
functionally the linear dense code, so this design would isolate threshold,
L1, and their interaction without changing parameter shapes.  It should not
reuse the opened development/confirmation condition for another selected
claim.

## Provenance

- Full-H200 frozen protocol:
  `/network/scratch/l/lia/skae-rebuttal/configs/allen_cahn_full_horizon_global_screen_20260719.json`,
  SHA-256 `d751d249ce4fc251182c8de6cde940c578b8545d184a9d0e3ead2f6a33f288dc`.
- Development selection and row evidence:
  `/network/scratch/l/lia/skae/allen_cahn_rebuttal_v2_20260719/full_horizon_global_summary/selection.json`
  (`2c8720da0eb65e6d11f90fd9e7d91dd957fcf9950f68d8e870f08d7c3ad1db8b`)
  and `rows.csv`
  (`9aa9f3d587026493a16c5920cb54b49e9a514d8b50c42a4feabf80de90478200`).
- Ten-seed architecture/capacity audit:
  `docs/figures/neurips_paper_2026/_data/allen_cahn_global_k_forecast_optimized_architecture_audit.json`,
  SHA-256 `f414cffce5c37144891e93292dbf9a6d0c66165170b9a20c4cbd3a7674ff2421`.
- Ten-seed statistics and provenance:
  `allen_cahn_global_k_forecast_optimized_statistics.json`
  (`554cfe69b0ef47261bd342ad869cd58a1dd5b8808f8bc397fee43ebd128581bb`)
  and `allen_cahn_global_k_forecast_optimized_provenance.json`
  (`00cee2ec300c4b7a70c0129a904bc3b2d0af396909b28c61b604d154fa1bbf07`).
- Focused architecture-screen rows and summary:
  `/network/scratch/l/lia/skae/allen_cahn_rebuttal_20260719/architecture_validation_screen/rows.csv`
  (`ecaa8892d1045049e449d33e26b648c42350be6c83fcb30b51dff4a4329632c4`)
  and `screen.json`
  (`f180d477bb3d25be914e163fc6564bcf1f57fe5ac3c3d3a68c6d27c3f1a60549`).
- Temporal-group screen decision:
  `/network/scratch/l/lia/skae/allen_cahn_sparse_forecast_20260719/forecast_loss_screen_summary/selection.json`,
  SHA-256 `92bdaad8a3fbd8ecff78dc79b0885e33be2a8e6e1b905914876571a13b569247`.
