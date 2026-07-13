# Lorenz-96 D128 Overcomplete Model Rows, 2026-06-24

## Concrete Results

Completed command:

```bash
uv run python -m experiments.run_suite --config configs/lorenz96_overcomplete_model_rows.yaml --benchmark lorenz96
uv run python scripts/analyze_lorenz96_overcomplete_model_rows.py --root results/lorenz96_overcomplete_model_rows_20260624
```

Execution used a CPU-only SLURM allocation. SLURM reported `Partition=long-cpu`, `ReqTRES=cpu=4,mem=24G,node=1`, and no GPU TRES. The config also set `device: "cpu"`, so no GPU was requested, allocated, or idled.

Condition: Lorenz-96 \(D=128\), \(F=8\), full observation, observation noise \(0.05\) times training-set channel standard deviation, 64 train / 16 validation / 16 test trajectories, seeds 0/1/2, training window 20, and \(d_z=512=4D\) for all neural rows. No neural row regularized \(K\); all neural rows used dense \(K\). Sparsity entered through latent L1 and/or sparse encoder nonlinearities.

Mean test NRMSE:

| model | H1 | H5 | H10 | H25 | H50 | H100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dmd | 0.2171 | 0.5534 | 0.7446 | 0.8965 | 0.9490 | 0.9741 |
| truncated_svd_dmd | 0.7106 | 0.7712 | 0.8242 | 0.8985 | 0.9473 | 0.9775 |
| dense_mlp_kae | 0.7595 | 0.8033 | 0.8413 | 0.9085 | 1.0176 | 1.0994 |
| sparse_mlp_l1_0.01 | 0.8320 | 0.8653 | 0.8953 | 0.9459 | 1.1592 | 5.5545 |
| sparse_mlp_l1_0.1 | 0.8237 | 0.8602 | 0.8910 | 0.9348 | 0.9807 | 1.0815 |
| lista_relu | 0.7008 | 0.7774 | 0.8364 | 0.9146 | 1.0720 | 1.2343 |
| lista_shrink | 0.6488 | 0.7516 | 0.8237 | 0.9147 | 1.0652 | 1.1639 |
| lista_sign_split | 0.7903 | 0.8369 | 0.8728 | 0.9460 | 1.1931 | 1.5965 |
| hyperlista | 0.2308 | 0.5515 | 0.6857 | 0.8112 | 1.1001 | 4.4427 |
| persistence | 0.2640 | 0.7425 | 1.0769 | 1.3434 | 1.3684 | 1.3919 |

At H100, DMD and truncated-SVD DMD were best. Among neural rows, `sparse_mlp_l1_0.1` was best at H50 and H100. Its H100 paired difference versus dense MLP KAE was \(-0.0179\) NRMSE, with 95% bootstrap CI \([-0.0380, 0.0055]\), so this run does not support a statistically secure H100 sparse-MLP improvement. At H50, the same row did improve dense MLP KAE by \(-0.0369\), CI \([-0.0444, -0.0291]\).

LISTA-family rows were not better than dense MLP at long horizons under this parity setup. `lista_shrink` and `lista_relu` were better than dense MLP at short horizons up to H10, but worse at H50 and H100. HyperLISTA was excellent at H1-H25 but diverged badly by H100.

## Interpretation

The overcomplete correction changes the paper-facing conclusion. The earlier undercomplete `d_z=64` pilot suggested clearer long-horizon latent-sparsity gains. With the required overcomplete latent \(d_z=512\), the strongest defensible result is narrower: stronger Sparse-MLP latent sparsity improves the matched dense MLP KAE at H50 and has a small, uncertain advantage at H100, while the tested LISTA configurations do not improve long-horizon forecasting.

Spectral radii moved closer to one for sparse/LISTA rows: dense MLP \(\rho(K)\approx1.070\), Sparse-MLP L1=0.1 \(\rho(K)\approx1.024\), LISTA-ReLU \(\rho(K)\approx1.013\), LISTA-shrink \(\rho(K)\approx1.011\). This stability diagnostic alone did not guarantee H100 accuracy.

## Project Implications

Do not use this overcomplete Lorenz-96 result to claim that LISTA broadly improves high-dimensional forecasting. It is useful as a stress test and as evidence that overcomplete sparse MLP regularization can help relative to dense MLP at moderate-to-long horizons, but DMD remains a stronger baseline in this fully observed state-space condition.

The next useful run should either tune LISTA thresholds/coefficients at \(d_z=512\), or move to partial/noisy observation histories where DMD is less advantaged by full-state Markov observations.

## Output Paths

- Raw metrics: `results/lorenz96_overcomplete_model_rows_20260624/results/raw_metrics.parquet`
- Wide NRMSE table: `results/lorenz96_overcomplete_model_rows_20260624/results/nrmse_wide_by_model.csv`
- Paired dense comparisons: `results/lorenz96_overcomplete_model_rows_20260624/results/paired_vs_dense_mlp_kae.csv`
- Diagnostics: `results/lorenz96_overcomplete_model_rows_20260624/results/diagnostics_summary.csv`
- Report: `results/lorenz96_overcomplete_model_rows_20260624/reports/overcomplete_model_rows_report.md`
- Figure: `results/lorenz96_overcomplete_model_rows_20260624/reports/figures/lorenz96_overcomplete_model_rows_nrmse.pdf`
- Config: `configs/lorenz96_overcomplete_model_rows.yaml`
- Analysis script: `scripts/analyze_lorenz96_overcomplete_model_rows.py`
