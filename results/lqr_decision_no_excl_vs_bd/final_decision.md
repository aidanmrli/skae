# LQR Decision

Decision stage: 2
System: lyapunov

Final choice: **no clear winner**
Fallback (simpler arm): **bd_c2**

## Primary Pairwise Metrics

| Metric (arm_a - arm_b) | Mean Diff | 95% CI | CI excludes 0 |
|---|---:|---:|:---:|
| m2_lqr_feasibility_rate | 0.0000 | [0.0000, 0.0000] | no |
| m3_closed_loop_stability_rate | 0.0000 | [0.0000, 0.0000] | no |
| m4_closed_loop_cost_reduction | -11.5848 | [-34.7394, 0.0088] | no |

## Robustness (M5)

- arm_a (bd_c2): 0.0
- arm_b (ah_prag_no_excl): 0.0

## Arm-Level Summary

| arm | n_runs | M2 mean | M3 mean | M4 mean |
|---|---:|---:|---:|---:|
| ah_prag_no_excl | 72 | 1.0000 | 1.0000 | 0.8230 |
| bd_c2 | 72 | 1.0000 | 1.0000 | -10.7638 |
