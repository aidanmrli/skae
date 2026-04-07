# LISTA Depth-Parity Plan vs `generic_sparse` (MLP)

Date: February 19, 2026  
Owner: SKAE experiments  
Status: Planning

## 1) Objective

Select a LISTA depth (`NUM_LOOPS`) that makes LISTA training dynamics and optimization behavior as close as possible to the `generic_sparse` MLP setup in `skae/config.py`.

Primary target:
1. Match optimization/training dynamics to `generic_sparse`.
2. Preserve or improve forecasting robustness.

Constraint for method design:
1. Do not rely on basin labels or known basin count for training-time decisions.

## 2) Why Depth Is Not the Only Lever

From `skae/config.py`, current preset differences are large:
1. `generic_sparse`: `MODEL_NAME=GenericKM`, `TARGET_SIZE=64`, `RECONST_COEFF=0.5`, `SPARSITY_COEFF=0.01`, MLP encoder `[64,64]`, bias on.
2. `lista` / `lista_nonlinear`: `MODEL_NAME=LISTAKM`, `TARGET_SIZE=2048`, `RECONST_COEFF=1.0`, `SPARSITY_COEFF=1.0`, homogeneous coordinates on.

Conclusion:
1. Depth-only comparisons are confounded unless non-depth settings are aligned first.

## 3) Parity Configuration (Proposed)

Create a new preset (name suggestion: `lista_parity_generic_sparse`) that copies `generic_sparse` optimization settings and only swaps encoder family to LISTA.

Keep fixed to `generic_sparse` values:
1. `TRAIN.LR = 1e-4`
2. `MODEL.TARGET_SIZE = 64`
3. `MODEL.RECONST_COEFF = 0.5`
4. `MODEL.SPARSITY_COEFF = 0.01`
5. `MODEL.RES_COEFF = 1.0`, `MODEL.PRED_COEFF = 0.0`
6. `MODEL.NORM_FN = "id"`
7. `MODEL.USE_HOMOGENEOUS = False`
8. `MODEL.ENCODER.USE_BIAS = True` (for closest affine behavior)
9. Keep decoder linear (`MODEL.DECODER.LAYERS = []`)

LISTA-specific defaults for parity sweep:
1. `MODEL.MODEL_NAME = "LISTAKM"`
2. `MODEL.ENCODER.LISTA.LINEAR_ENCODER = False` (use nonlinear front-end for closer MLP-like parameterization)
3. `MODEL.ENCODER.LAYERS = [64, 64]` (match MLP width/depth)
4. `MODEL.ENCODER.LISTA.FINAL_OP = "shrink"`
5. Start with moderate shrink threshold and tune lightly after depth selection:
   - initial `LISTA.ALPHA = 0.1` (or small pilot over `{0.05, 0.1, 0.2}`)
6. Let LISTA use internally computed Lipschitz scale at model init.

## 4) Depth Candidates and Expectation

Sweep:
1. `NUM_LOOPS in {0, 1, 2, 3}`

Expected shape:
1. `0-1` loops should be closest to feedforward MLP dynamics.
2. `2-3` may help fit quality but can drift toward iterative LISTA dynamics.
3. Choose smallest depth that meets closeness and robustness criteria.

## 5) Closeness-to-`generic_sparse` Metrics

Use the same systems/seeds for both models and compare depth-wise against `generic_sparse`.

### A. Training-dynamics similarity
1. Loss trajectory distance: `total`, `alignment`, `reconstruction`, `sparsity`.
2. Gradient/update scale distance (encoder and full model parameter groups).
3. Early-training stability indicators:
   - spike count in loss,
   - NaN/Inf incidence,
   - variance across seeds.

### B. Forecast behavior similarity
1. `H1000 no-reencode`, `H1000 every-step`, `H1000 best-periodic`.
2. Good-system count (`H1000 best-periodic < 10`).
3. Catastrophic-system count (`H1000 best-periodic >= 1000`).
4. Best-period mode profile distance to `generic_sparse`.

### C. Composite score (for ranking depths)
Define:
1. `D_train`: normalized distance on training curves/stability.
2. `D_forecast`: normalized distance on H1000 metrics and mode profile.
3. `D_robust`: normalized distance on seed robustness.

Rank by:
1. `Score = 0.45 * D_train + 0.35 * D_forecast + 0.20 * D_robust`

Hard gates before score:
1. No increase in catastrophic-system count vs `generic_sparse` by more than 1.
2. No severe seed instability regression.

## 6) Decision Rule for `depth_star_parity`

1. Filter depths passing hard gates.
2. Pick minimum composite `Score`.
3. Tie-breakers:
   1. Lower catastrophic count.
   2. Lower `NUM_LOOPS` (prefer simpler dynamics).
   3. Better median `H1000 best-periodic`.

## 7) Execution Plan

### Phase A: Small parity pilot
1. Systems: 4 representative systems (easy + hard + multiwell).
2. Seeds: `{0,1,2}`.
3. Depths: `{0,1,2,3}`.
4. Output: quick ranking and sanity-check of chosen metrics.

### Phase B: Full parity benchmark
1. Systems: canonical 23-system set.
2. Seeds: `{0,1,2}`.
3. Depths: shortlist from Phase A (typically top 2-3).
4. Output: final `depth_star_parity`.

### Phase C: Lock depth, then minor threshold tuning
1. Fix `NUM_LOOPS=depth_star_parity`.
2. Small `ALPHA` sweep for tail reduction with minimal dynamics drift.

## 8) Reporting Requirements

For each phase, report in this order:
1. Concrete results.
2. Context of those results.
3. Interpretation.
4. Project implications.
5. Next steps.

Update:
1. `docs/EXPERIMENTS.md` Current Status Summary.
2. `docs/EXPERIMENTS.md` Outstanding problems.
3. `docs/EXPERIMENTS.md` Queue Status.
4. Relevant experiment log entry.

## 9) Immediate Next Actions

1. Add `lista_parity_generic_sparse` preset in `skae/config.py`.
2. Add a sweep script for parity depth (`NUM_LOOPS` grid above).
3. Ensure collectors produce side-by-side comparison tables vs `generic_sparse`.
4. Launch Phase A pilot and compute `depth_star_parity`.
