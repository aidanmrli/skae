# Lorenz-96 D128 Overcomplete Model Rows, Long-Budget Run, 2026-06-24

## Concrete Results

Completed command:

```bash
uv run python -m experiments.run_suite --config configs/lorenz96_overcomplete_model_rows_long.yaml --benchmark lorenz96
uv run python scripts/analyze_lorenz96_overcomplete_model_rows.py --root results/lorenz96_overcomplete_model_rows_long_20260624
```

Execution used a CPU-only SLURM allocation on `long-cpu`; SLURM reported `ReqTRES=cpu=4,mem=24G,node=1` and no GPU TRES. The config used `device: "cpu"`, so no GPU was requested or idled.

Condition: Lorenz-96 \(D=128\), \(F=8\), full observation, observation noise \(0.05\), 64 train / 16 validation / 16 test trajectories, seeds 0/1/2, training window 20, \(d_z=512=4D\), dense learned \(K\), and no \(K\) regularization. Neural rows were trained with a maximum of 2,000 minibatch optimizer steps and patience 200. This is the fair long-budget replacement for the earlier 120-step smoke run.

Mean test NRMSE:

| model | H1 | H5 | H10 | H25 | H50 | H100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DMD | 0.2171 | 0.5534 | 0.7446 | 0.8965 | 0.9490 | 0.9741 |
| truncated-SVD DMD | 0.7106 | 0.7712 | 0.8242 | 0.8985 | 0.9473 | 0.9775 |
| dense MLP KAE | 0.4554 | 0.6096 | 0.7082 | 0.8175 | 0.9131 | 1.0081 |
| sparse MLP L1=0.01 | 0.4906 | 0.6253 | 0.7131 | 0.8130 | 0.9823 | 3.8219 |
| sparse MLP L1=0.1 | 0.4781 | 0.6202 | 0.7098 | 0.8103 | 0.9083 | 1.0107 |
| LISTA-ReLU | 0.3858 | 0.5527 | 0.6511 | 0.7645 | 0.9642 | 1.3711 |
| LISTA-shrink | 0.4190 | 0.6005 | 0.7034 | 0.8086 | 0.9184 | 1.0219 |
| LISTA-sign-split | 0.3974 | 0.5563 | 0.6513 | 0.7565 | 0.8832 | 0.9525 |
| HyperLISTA | 0.2185 | 0.5141 | 0.6345 | 0.7673 | 1.1069 | 2.9002 |
| persistence | 0.2640 | 0.7425 | 1.0769 | 1.3434 | 1.3684 | 1.3919 |

The main change from the 120-step run is that LISTA-sign-split becomes the best H50/H100 model. Selected paired differences:

| model | baseline | horizon | diff model-baseline | 95% bootstrap CI |
| --- | --- | ---: | ---: | ---: |
| LISTA-sign-split | dense MLP KAE | 50 | -0.0299 | [-0.0419, -0.0180] |
| LISTA-sign-split | dense MLP KAE | 100 | -0.0556 | [-0.0667, -0.0452] |
| LISTA-sign-split | DMD | 50 | -0.0658 | [-0.0797, -0.0512] |
| LISTA-sign-split | DMD | 100 | -0.0216 | [-0.0295, -0.0139] |
| LISTA-sign-split | truncated-SVD DMD | 50 | -0.0641 | [-0.0784, -0.0495] |
| LISTA-sign-split | truncated-SVD DMD | 100 | -0.0250 | [-0.0339, -0.0165] |

Sparse MLP L1=0.1 improved at H50 but not H100 relative to dense MLP. LISTA-shrink improved substantially over the short run but remained slightly worse than dense MLP at H100. LISTA-ReLU and HyperLISTA still became unstable at H100.

## Training Budget

Most rows did not consume the full 2,000-step maximum because early stopping was enabled. Actual optimizer steps by row/seed were:

- dense MLP KAE: 1129, 1463, 1408
- sparse MLP L1=0.01: 2000, 2000, 1911
- sparse MLP L1=0.1: 2000, 2000, 1906
- LISTA-ReLU: 1884, 1155, 1024
- LISTA-shrink: 632, 1156, 1017
- LISTA-sign-split: 2000, 1154, 997
- HyperLISTA: 596, 1154, 630

The fair comparison is therefore "same max budget and same early-stopping rule," not identical realized step count.

## Interpretation

The earlier 120-step run was too short for this overcomplete comparison. With a longer fair budget, dense MLP improved strongly and LISTA-sign-split improved enough to beat dense MLP, DMD, and truncated-SVD DMD at H50/H100 in paired trajectory-level comparisons. This supports using long-budget overcomplete LISTA-sign-split as the main Lorenz-96 high-dimensional forecasting row.

The result is still specific: it uses full-state noisy Lorenz-96, \(d_z=512\), a 20-step training window, and dense \(K\). It does not show that all LISTA variants work, nor that sparse MLP is reliably better at H100.

## Output Paths

- Raw metrics: `results/lorenz96_overcomplete_model_rows_long_20260624/results/raw_metrics.parquet`
- Wide NRMSE table: `results/lorenz96_overcomplete_model_rows_long_20260624/results/nrmse_wide_by_model.csv`
- Paired vs dense: `results/lorenz96_overcomplete_model_rows_long_20260624/results/paired_vs_dense_mlp_kae.csv`
- Selected paired differences: `results/lorenz96_overcomplete_model_rows_long_20260624/results/selected_paired_differences.csv`
- Short-vs-long comparison: `results/lorenz96_overcomplete_model_rows_long_20260624/results/short_vs_long_nrmse_delta.csv`
- Figure: `results/lorenz96_overcomplete_model_rows_long_20260624/reports/figures/lorenz96_overcomplete_model_rows_nrmse_zoom.pdf`
- Config: `configs/lorenz96_overcomplete_model_rows_long.yaml`
