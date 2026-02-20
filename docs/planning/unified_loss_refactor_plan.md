# Refactor Plan: Pure Aggregation Loss API with External Rollout

## Summary
Refactor training so rollout/trajectory construction happens outside the loss function, and the model loss only aggregates precomputed tensors.

Canonical loss interface:
- Required:
  - `x_pred` `[B, H, ...]` (predicted `x_{t+1...t+H}`)
  - `x_true` `[B, H, ...]` (ground truth `x_{t+1...t+H}`)
- Optional:
  - `x0` `[B, ...]` (ground truth `x_t`)
  - `z0` `[B, ...]` (encoded `z_t`)
  - `z_pred` `[B, H, Dz]`
  - `z_true` `[B, H, Dz]`
  - `extras` `dict[str, Tensor]` for structured terms

Loss remains: prediction + alignment + reconstruction + sparsity.
Loss function is pure aggregator (no encode/decode/rollout inside).

## Final API and Contracts

### 1. Unified loss signature (model-level)
`loss(x_pred, x_true, x0=None, z0=None, z_pred=None, z_true=None, extras=None, step=0)`

Behavior:
1. Validate shapes and horizon consistency (`H >= 1`).
2. Compute terms only from provided tensors.
3. Aggregate weighted total from config coefficients.
4. Return `(total_loss, metrics)`.

Notes:
- Pairwise is `H=1`.
- Multistep is `H>1`.
- No branchy "pairwise vs sequence" loss paths.

### 2. Required precomputations outside loss
Training step prepares:
1. `x_pred`, `z_pred` from rollout engine (discrete K application only).
2. `x_true`, `z_true` sliced/encoded from data windows.
3. `x0`, `z0` from initial state.
4. Structured extras (if model uses them), e.g. per-step basin tensors or precomputed penalties.

### 3. Scaling rule (locked)
For all horizon-dependent terms:
- `num_k_applications = H`
- `sequence_term_scale = 1.0 / H`

Apply to each horizon-based term:
- prediction
- alignment
- reconstruction over predicted horizon
- sparsity over predicted horizon
- structured horizon penalties (exclusivity/temporal/etc., if present)

Initial-state-only terms (if any) are not horizon-scaled unless explicitly defined as horizon terms.

## Implementation Changes

### A. `skae/model.py`
1. Introduce pure aggregation helper in base class:
   - `_aggregate_losses_from_tensors(...)`
2. Replace existing mixed compute+loss methods with:
   - aggregator `loss(...)` using the new signature.
3. Remove ODE-specific methods and dependency usage.
4. Remove seq-8 hardcoded scaling.
5. For Structured model:
   - keep structured term math but consume `extras`/precomputed tensors instead of recomputing rollouts inside loss.
6. Keep discrete rollout utilities outside `loss` (model helper methods are okay if called from train step, not from loss).

### B. `tools/train.py`
1. Training pipeline always:
   - samples sequence windows,
   - computes rollout tensors outside loss,
   - calls unified `loss(...)`.
2. Remove mode flags and mode branching.
3. Use `--sequence_length` as the only horizon control:
   - `sequence_length=1` equals former pairwise behavior.
4. Update logs/help strings accordingly.

### C. Config + compatibility
1. Remove active dependence on separate pairwise/sequence mode booleans.
2. Keep loader compatibility for legacy config keys by ignoring them.
3. Keep coefficient semantics unchanged.

### D. Dependency cleanup
1. Remove `torchdiffeq` from runtime imports and `pyproject.toml`.
2. Refresh `uv.lock`.

### E. Scripts/docs
1. Update scripts that pass old mode flags to use only `--sequence_length`.
2. Update docs language from ODE/mode-specific wording to unified horizon-based training.
3. Update experiment notes to describe horizon scaling by K applications.

## Loss Term Definitions (Pure Aggregator)

Given inputs:
- `x_pred`, `x_true` shape `[B,H,Dx]`
- `z_pred`, `z_true` shape `[B,H,Dz]`
- optional `x0`, `z0`

Compute:
1. `prediction_loss`: mean over batch/time of `||x_pred - x_true||` (existing norm convention preserved).
2. `alignment_loss`: mean over batch/time of `||z_pred - z_true||` (requires both latents).
3. `reconst_loss`: from precomputed recon tensors in `extras` (pure rule), or explicitly passed as `extras["reconst_error"]`.
4. `sparsity_loss`: from precomputed latent tensors (`extras["z_for_sparsity"]`), default to `z_pred` and optionally include `z0`.
5. structured losses (if any): read from `extras` tensors, then apply configured warmup/weights.

Then multiply horizon-based terms by `1/H`, combine with configured coefficients, return metrics.

## Testing Plan

### 1. Unit tests (`tests/test_model.py`)
1. Interface validation:
   - missing required latent tensors when alignment enabled raises clear error.
2. Scaling tests:
   - `H=1,2,8` verifies exact `1/H` behavior.
3. Pure-loss test:
   - monkeypatch encode/decode/rollout to fail; loss still works from provided tensors.
4. Structured extras:
   - verify extras-driven structured terms included and weighted correctly.

### 2. Training tests (`tests/test_train.py`)
1. End-to-end with `sequence_length=1` (former pairwise).
2. End-to-end with `sequence_length=8`.
3. CLI no-mode-flags behavior.
4. Metrics presence/shape consistency across horizons.

### 3. Regression tests
1. Remove references to seq-8 special scaling.
2. Ensure no ODE symbols/imports remain in model training path.
3. Smoke run of one generic and one structured config.

## Acceptance Criteria
1. One canonical loss API (`loss(x_pred, x_true, x0=None, z0=None, z_pred=None, z_true=None, extras=None)`).
2. Loss is pure aggregator: no rollout/encode/decode calls inside.
3. Pairwise behavior achieved by horizon `H=1`.
4. All horizon-dependent terms scaled by `1/H`.
5. No ODE training code/dependency remains.
6. Updated tests pass.

## Assumptions and Defaults
1. Alignment is required for the standard objective; if `z_pred/z_true` absent, alignment term is skipped only when coefficient is zero.
2. Reconstruction and structured-specific inputs are passed via `extras` to keep loss pure.
3. Existing norm forms are preserved unless explicitly changed during implementation.
