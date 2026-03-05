# LISTA-Stack Forecasting Improvement Plan

Date: 2026-03-04

## Goal

- Improve LISTA-stack (`LISTAKM`) forecasting (especially long-horizon rollouts) while preserving basin-support alignment.
- Target `sparsity_ratio` (inactive latent dims) in the `0.7–0.9` range during steady training.
- Close the forecasting gap vs `generic_sparse` on benchmark systems (start with Duffing): aim for `<= 2x` on `H500/H1000` best-periodic, with comparable robustness across seeds.

## Current Evidence (Duffing)

- Controlled encoder comparison (Experiment K; `L=8`, `50k`, 3 seeds) shows LISTA remains substantially worse than `generic_sparse` even after matching loss coefficients.
- LISTA codes are much denser:
  - LISTA: `sparsity_ratio ~ 0.18–0.23`
  - `generic_sparse`: `sparsity_ratio ~ 0.78–0.80`
- A plausible root cause is threshold scaling: LISTA shrinkage uses a threshold of `alpha / L`, and in low-dimensional observations with highly overcomplete dictionaries, the computed `L` can be large, making `alpha / L` tiny. The result is many small-but-nonzero coefficients rather than clean zeros.

## Immediate Change (Now Default)

- Enforce **ReLU as the final operation** in the LISTA loop: the last step returns `ReLU(shrink(.))`.
  - Motivation: match `generic_sparse`’s nonnegative latent codes (`LAST_RELU=True`) and increase exact zeros (negative coefficients are clamped to 0).

## Hypotheses To Test (No Basin Labels Required)

- **H1: Sparsity is threshold-limited**. `alpha / L` is too small, so shrinkage rarely produces zeros. Increasing the effective threshold yields `sparsity_ratio` in `0.7–0.9` and improves forecasting.
- **H2: Encoder capacity is limiting**. With `NUM_LOOPS=5` and tied parameters, LISTA cannot represent the x→z map that best supports multi-step Koopman rollouts.
- **H3: Decoder/dictionary constraints matter**. Per-forward dictionary normalization (and lack of bias/homogeneous coordinate) may restrict reconstruction quality and/or distort the optimization landscape for LISTA-stack.
- **H4: We need stable supports, not just zeros**. Forecasting improves when supports are temporally stable and consistent across rollouts (support jitter makes K-fitting and rollouts brittle).

## Metrics (Training-Time Label-Free)

- Forecasting:
  - Quick eval best (during training).
  - Standardized evaluation: best-periodic `H100/H500/H1000` (and distribution of best-period modes).
- Sparsity:
  - `sparsity_ratio`, support size distribution, and magnitude histogram of `|z|`.
- Support stability:
  - Within-trajectory support overlap across time (e.g., Jaccard of top-k or thresholded supports).
- Stability:
  - Spectral radius of `K` and frequency of catastrophic rollouts.
- Evaluation-only diagnostics (allowed on benchmarks):
  - Separability by known basin labels / known basin counts, but do not use in training-time method design.

## Experiment Queue (Start With Duffing; `L=8`, `50k`, 3 seeds)

### 0) Post-Change Baseline (ReLU-Final LISTA)

- Re-run the LISTA baseline (same as Experiment K, but with enforced ReLU-final).
- Record: forecasting metrics, `sparsity_ratio`, and best-period mode distribution.
- Purpose: establish the new reference point after the nonlinearity change.

### 1) LISTA Threshold (`alpha`) Sweep (Fast Gate Then Full)

- Sweep `lista_alpha` over a coarse grid (example): `{0.1, 0.3, 1.0, 3.0, 10.0}`.
- Keep loss coefficients fixed (matched to `generic_sparse`) to isolate threshold effects.
- Run a short gate (`10k`) to filter obviously-bad settings, then full `50k` on survivors.
- Success criteria:
  - `sparsity_ratio` enters `0.7–0.9`.
  - Forecasting does not collapse at long horizons (especially `H500/H1000`).

### 2) Sparsity Penalty Sweep (Given a Good `alpha`)

- For the best `alpha` values, sweep `SPARSITY_COEFF` (example): `{0.005, 0.01, 0.02, 0.05}`.
- Watch for the failure mode “many tiny nonzeros” vs “clean zeros,” and how that maps to rollouts.

### 3) LISTA Depth / Capacity Sweep

- Sweep `NUM_LOOPS` (example): `{0, 2, 5, 10, 20}`.
- If additional loops help:
  - Consider untying `S` per loop (more capacity).
  - Consider per-loop step sizes / per-loop thresholds (more adaptive inference).

### 4) Adaptive Thresholding (Reduce Manual Tuning)

- Try `ENCODER_TYPE=hyperlista` as a first step toward instance-adaptive thresholds.
- If HyperLISTA helps, prefer it as the default “adaptive LISTA” baseline for forecasting.
- If not, implement a lightweight variant:
  - Learnable per-dimension thresholds `theta_i`.
  - Or learnable per-loop thresholds `theta_k`.

### 5) Input Map (`We`) Variants and Activations

- Compare `LINEAR_ENCODER=True` (analytic-style) vs `LINEAR_ENCODER=False` (MLP `We`).
- If using MLP `We`, sweep activation: `{relu, gelu, tanh}`.
- Optionally test `LAST_RELU=True` for the MLP `We` output (in addition to final ReLU in LISTA).

### 6) Decoder / Dictionary Variants

- Toggle `DECODER.AFFINE_BIAS` and/or `USE_HOMOGENEOUS` for LISTA-stack (keep the rest fixed).
- If normalization seems restrictive, test a “gain” parameter per atom (scale columns without changing directions) while keeping normalization.

### 7) Training/Optimization Stabilizers (If Needed)

- Separate optimizer parameter groups / LRs for `{K, encoder, dict}`.
- Gradient clipping (especially for `S` and `K`).
- Soft spectral-radius penalty when `SR > 1` (only if instability dominates).

## Interpretation Rules

- Do not treat lower training loss as success; require:
  - forecasting metrics + robustness across seeds, and
  - `sparsity_ratio` in target band, and
  - reduced reliance on very short reencoding periods (best-period collapse is a warning sign).
- Prefer interventions that improve long-horizon robustness first; quick-best can be tuned after.

## Next Steps

- Run the post-change baseline, then the `alpha` sweep (short gate first).
- Only after achieving the target `sparsity_ratio` should we invest in bigger capacity/decoder changes.

