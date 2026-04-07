# HyperLISTA Stabilization Implementation Plan

1. Add HyperLISTA stabilization config knobs in [config.py](/home/mila/l/lia/skae/skae/config.py).
- Add `C_THETA_MIN` (e.g. `1e-6`).
- Add `CONSTRAIN_C_THETA` (`True`).
- Add `PINV_CACHE_MODE` (`"none"` default for now).
- Keep `USE_SUPPORT_SELECTION` and `USE_MOMENTUM` configurable from CLI.
- Add `SPARSITY_TARGET` and `SPARSITY_TARGET_COEFF` (default `0.8`, `0.0`).

2. Reparameterize `c_theta` to be strictly positive in [model.py](/home/mila/l/lia/skae/skae/model.py).
- Replace unconstrained `encoder.c_theta` with raw param + transform: `softplus(raw) + C_THETA_MIN`.
- Use transformed value wherever `theta` is computed.

3. Fix `D_pinv` handling in [model.py](/home/mila/l/lia/skae/skae/model.py).
- Remove `data_ptr`-based stale-cache invalidation.
- Recompute `pinv(D)` each forward pass (first safe version).
- Keep caching optimization for later only if we add robust invalidation.

4. Make support-selection safer for recovery runs in [model.py](/home/mila/l/lia/skae/skae/model.py).
- Keep current support-selection code path.
- Run Queue-5b defaults with `USE_SUPPORT_SELECTION=False`.
- Leave support-selection as optional ablation flag.

5. Add sparsity-target penalty in unified loss in [model.py](/home/mila/l/lia/skae/skae/model.py).
- Compute current `sparsity_ratio` (already available).
- Add loss term: `SPARSITY_TARGET_COEFF * (sparsity_ratio - SPARSITY_TARGET)^2`.
- Log `sparsity_target_loss` metric.

6. Expose new CLI overrides in [train.py](/home/mila/l/lia/skae/tools/train.py).
- `--sparsity_target`, `--sparsity_target_coeff`.
- `--hyperlista_use_ss`, `--hyperlista_use_momentum`, `--hyperlista_constrain_c_theta`, `--hyperlista_c_theta_min`.

7. Add tests in [tests](/home/mila/l/lia/skae/tests).
- HyperLISTA `c_theta` stays positive after optimizer steps.
- `pinv` refresh behavior no longer tied to `data_ptr`.
- Sparsity-target loss activates only when coeff > 0 and is logged.

8. Create Queue-5b scripts from Queue-5 templates in [scripts](/home/mila/l/lia/skae/scripts).
- Run a 10k quick sweep first.
- Arms:
- A: constrained `c_theta`, support-selection off, momentum off.
- B: constrained `c_theta`, support-selection off, momentum on.
- C: constrained `c_theta`, support-selection on, momentum on.
- Keep `sp={0.004,0.006}`, `loops=1`, 3 seeds.

9. Add collector and ranking updates (same summary/Pareto pipeline).
- Reuse [summarize_encoder_comparison.py](/home/mila/l/lia/skae/tools/summarize_encoder_comparison.py) and [compute_pareto_frontier.py](/home/mila/l/lia/skae/tools/compute_pareto_frontier.py).
- Compare Queue-5b directly against Queue-4 anchors and `generic_sparse`.

10. Update documentation in [EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md).
- Record code changes, rationale, and Queue-5b acceptance criteria:
- At least one in-band arm (`0.7–0.9`).
- Long-horizon materially better than current Queue-5.
- No catastrophic outlier behavior.
