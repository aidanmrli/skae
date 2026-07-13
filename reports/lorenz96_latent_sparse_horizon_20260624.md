# Lorenz-96 D128 Latent-Sparsity Horizon Sweep, 2026-06-24

## Concrete Results

Completed run:

```bash
uv run python -m experiments.run_suite --config configs/lorenz96_latent_sparse_horizon.yaml --benchmark lorenz96
uv run python scripts/analyze_lorenz96_latent_sparse_horizon.py --root results/lorenz96_latent_sparse_horizon_20260624
```

Execution used a SLURM interactive CPU allocation on `cn-m002`. No failures were recorded in `results/lorenz96_latent_sparse_horizon_20260624/results/failures.json`.

The benchmark used Lorenz-96 with \(D=128\), \(F=8\), full observation, additive observation noise at 0.05 times each channel's training standard deviation, 64 training trajectories, 16 validation trajectories, and 16 test trajectories for each seed. Seeds were 0, 1, and 2. The dense and sparse models used the same tanh MLP encoder/decoder, latent dimension 64, and dense learned Koopman matrix \(K\). The sparse arms did not regularize \(K\). They used an L1 penalty on rolled latent activations. Training rollout/window lengths were 10, 20, and 40 stored steps. Test rollout horizons were 1, 5, 10, 25, 50, and 100.

Validation-selected sparse coefficients improved test NRMSE relative to the matched dense KAE:

| train window | selected latent L1 | H25 delta | H50 delta | H100 delta |
|---:|---:|---:|---:|---:|
| 10 | 0.01 | -0.0081 | -0.0104 | -0.0059 |
| 20 | 0.01 | -0.0019 | -0.0110 | -0.0244 |
| 40 | 0.10 | -0.0176 | -0.0119 | -0.0174 |

Entries are paired sparse-minus-dense NRMSE, so negative values favor sparse. The 95% trajectory-level bootstrap confidence intervals for these selected long-horizon deltas exclude zero in favor of sparse for H25, H50, and H100.

The full sparsity curve showed the strongest long-horizon gains at latent L1 coefficient 0.10:

| train window | horizon | sparse NRMSE | dense NRMSE | paired delta | percent change |
|---:|---:|---:|---:|---:|---:|
| 10 | 100 | 1.0686 | 1.2037 | -0.1350 | -11.2% |
| 20 | 100 | 0.9886 | 1.0799 | -0.0913 | -8.5% |
| 40 | 100 | 0.9875 | 1.0049 | -0.0174 | -1.7% |

Window length mattered. Moving from 10 to 20 steps improved the dense KAE H100 NRMSE from 1.2037 to 1.0799 and the best sparse H100 NRMSE from 1.0686 to 0.9886. A 40-step window gave the best absolute H100 sparse result, 0.9875, but the 20-step window gave the best H25 and H50 sparse results among neural models.

The no-K-regularization caveat is important. Effective density of \(K\) at the \(10^{-3}\max|K|\) threshold stayed high, around 0.93 to 0.94 for the 20-step sparse and dense models. The result supports latent sparsity improving forecasting and rollout stability; it does not support a claim that an elementwise sparse \(K\) caused the improvement.

Classical DMD remained a strong baseline in this focused condition. At H100, DMD achieved mean NRMSE 0.9741 and truncated-SVD DMD achieved 0.9775, both slightly below the best neural sparse value 0.9875. The result therefore supports "latent sparsity improves the matched KAE forecast baseline on high-dimensional noisy Lorenz-96", not "SKAE beats all baselines".

## Interpretation

The earlier few-trajectory run was a low-data stress test. This run uses the full 64-trajectory training split and is a better candidate for the high-dimensional forecasting statement. The evidence is consistent with the user's hypothesis that longer multistep windows stabilize learned rollouts: the 20-step and 40-step windows substantially reduce long-horizon error compared with the 10-step window.

The latent L1 penalty has a regularizing effect even with dense \(K\). Spectral radii moved closer to the unit circle with longer training windows and moderate or strong latent sparsity. For example, mean dense spectral radius was about 1.074 for a 10-step window, 1.034 for a 20-step window, and 1.016 for a 40-step window. Sparse models with coefficient 0.10 had mean spectral radius about 1.023, 1.020, and 1.025 for windows 10, 20, and 40.

## Project Implications

This result can be used as an extension benchmark showing that sparse latent KAEs remain useful beyond low-dimensional state spaces and that the advantage is visible under controlled observation noise. The strongest defensible sentence is:

"On a noisy \(D=128\) Lorenz-96 forecasting benchmark, latent sparsity improved matched dense-KAE long-rollout NRMSE under all tested training-window lengths, with validation-selected sparse models reducing 100-step NRMSE by 0.5%, 2.3%, and 1.7% for 10-, 20-, and 40-step training windows, respectively."

Do not describe this as evidence for sparse \(K\), because \(K\) was not regularized and remained effectively dense. Do not claim superiority over DMD in this condition.

## Next Steps

1. Promote the 20-step window as the default for this high-dimensional Lorenz extension, with 40-step as a robustness check.
2. Add a validation-selection table to the paper appendix if this benchmark becomes paper-facing.
3. Run a second condition with stronger noise or partial observation if the paper needs a setting where DMD is less competitive.
4. If the scientific claim must be "sparse operator", run a separate explicitly labeled K-regularized ablation; keep it separate from the latent-sparsity claim.

## Output Paths

- Raw metrics: `results/lorenz96_latent_sparse_horizon_20260624/results/raw_metrics.parquet`
- Paired deltas: `results/lorenz96_latent_sparse_horizon_20260624/results/paired_sparse_minus_dense_nrmse.csv`
- Validation-selected sparse results: `results/lorenz96_latent_sparse_horizon_20260624/results/validation_selected_sparse_test_nrmse.csv`
- Absolute NRMSE summary: `results/lorenz96_latent_sparse_horizon_20260624/results/absolute_nrmse_by_model.csv`
- Spectral summary: `results/lorenz96_latent_sparse_horizon_20260624/results/spectral_summary.csv`
- Figure: `results/lorenz96_latent_sparse_horizon_20260624/reports/figures/lorenz96_latent_sparse_horizon_deltas.pdf`
- Config: `configs/lorenz96_latent_sparse_horizon.yaml`
- Analysis script: `scripts/analyze_lorenz96_latent_sparse_horizon.py`
