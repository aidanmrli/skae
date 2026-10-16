# Bundled evidence

Results of the paper's runs. Commands are `uv run python -m paper <command>`; `evidence` runs every producer.

| File | Contents | Produced by | Used by |
|---|---|---|---|
| `forecasting_multibasin.csv` | test MSE per multibasin run and horizon | `collect` | Tables 1, 8, 12–14; Figure 2a |
| `forecasting_dysts.csv` | test MSE per Dysts run and horizon | `collect` | Tables 1, 15–17; Figure 2b |
| `forecasting_window64.csv` | test MSE of the 64-state-window runs | `collect` | Table 8 |
| `forecasting_sparsity_ablation.csv` | test MSE of the sparsity-ablation encoders | `collect` | Table 6 |
| `baselines_multibasin.csv` | DMD and EDMD test MSE, multibasin | `baselines` | Tables 1, 12–14 |
| `baselines_dysts.csv` | DMD and EDMD test MSE, Dysts | `baselines` | Tables 1, 15–17 |
| `support_transfer.csv` | support–basin alignment per multibasin run | `analyze support-transfer` | Tables 1, 9–11; Figure 5b |
| `support_sensitivity.json.gz` | the alignment over support and Jaccard thresholds | `analyze support-sensitivity` | Table 11 |
| `reencoding.csv` | test MSE per re-encoding condition | `analyze reencoding` | Table 7 |
| `interventions.csv.gz` | MSE under support interventions (LISTA, Local-linear gates, seed 0) | `analyze interventions` | Table 2; Figure 4 |
| `sparsity_mechanisms.json` | fractions of inactive codes and zero hidden activations | `analyze sparsity-mechanisms` | Table 6 |
| `error_growth.json` | squared forecast error over time per Dysts system | `analyze error-growth` | Figure 3 |

`SHA256SUMS` covers every file here but this README; `rebuild` checks it, and `evidence` writes one for each
new directory.
