# Lorenz-96 D=128 noisy sparse-K forecasting result

Date: 2026-06-24.

## Concrete results

I ran Lorenz-96 at `D=128`, forcing `F=8`, full observation, observation noise `0.05 * train-channel std`, seeds `0,1,2`, and train sizes `4`, `12`, and `24` trajectories. Test sets were fixed within each seed/noise condition across train-size arms. Rollout horizons were `1,5,10,25,50,100`.

The most useful sparse setting was `K` L1 coefficient `10.0`. At horizon 100:

| Train trajectories | Dense KAE NRMSE | Sparse KAE NRMSE | Paired sparse-dense delta | 95% bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| 4 | 1.211953 | 1.002905 | -0.209048 | [-0.222208, -0.196914] |
| 12 | 1.180356 | 0.999558 | -0.180799 | [-0.192886, -0.169413] |
| 24 | 1.143666 | 1.001100 | -0.142567 | [-0.152456, -0.131943] |

At horizon 50, sparse KAE also improved over dense KAE:

| Train trajectories | Dense KAE NRMSE | Sparse KAE NRMSE | Paired sparse-dense delta | 95% bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| 4 | 1.150244 | 0.996751 | -0.153492 | [-0.164139, -0.142805] |
| 12 | 1.124591 | 0.999936 | -0.124655 | [-0.138679, -0.111132] |
| 24 | 1.076546 | 1.000553 | -0.075993 | [-0.085861, -0.065943] |

Effective K density at threshold `1e-3 * max(|K|)` fell from dense KAE densities of `0.926`, `0.904`, and `0.882` for train sizes `4`, `12`, and `24` to sparse densities of `0.119`, `0.016`, and `0.016`.

## Context

This is not the full requested benchmark grid. It is a targeted high-dimensional noisy forecasting benchmark designed to test whether sparsity helps when the dense Koopman operator can overfit or drift over long rollouts.

The result is strongest for long horizons. At horizons `1`, `5`, and often `10`, dense and sparse KAE are similar, with dense sometimes slightly better. The sparse advantage appears in long free-running forecasts.

## Interpretation

This supports a narrow claim: **on noisy D=128 Lorenz-96, strong Koopman-matrix sparsity improves long-horizon forecasting relative to a dense KAE with the same encoder/decoder and latent dimension.**

It does not support the stronger claim that SKAE dominates all baselines. Classical DMD/POD-DMD remains competitive and is better in several train-12/train-24 mid-horizon settings. However, in the lowest-data condition (`4` train trajectories), sparse KAE is the best recorded method at horizons `50` and `100`.

## Project implications

This gives a defensible high-dimensional, noisy, long-rollout result for the paper, provided it is framed as a sparse-versus-dense Koopman ablation rather than a universal benchmark win.

The evidence is useful because it isolates the causal factor of `K` sparsity: same generated data, same splits, same model capacity, same seeds, same rollout protocol, and paired trajectory-level bootstrap differences.

## Next steps

1. Add a clean `noise=0.0` control to show the sparse benefit is a noisy/regularized regime effect.
2. Add `D=256` with the same selected grid: dense KAE versus `lambda_K=10`.
3. Re-run the selected setting with a stronger neural baseline only if needed; otherwise use DMD/POD-DMD as honest classical references.
4. Move the concise claim and table into `docs/neurips_sparse_koopman_multibasin.tex` only if senior coauthors want this as a paper-facing benchmark extension.

Artifacts:

- Raw metrics: `results/lorenz96_highdim_sparse_20260624/results/raw_metrics.parquet`
- Summary metrics: `results/lorenz96_highdim_sparse_20260624/results/summary_metrics.csv`
- Paired sparse-dense deltas: `results/lorenz96_highdim_sparse_20260624/results/paired_sparse_minus_dense_nrmse.csv`
- Report: `results/lorenz96_highdim_sparse_20260624/reports/final_report.md`
- Figures: `results/lorenz96_highdim_sparse_20260624/reports/figures/`
