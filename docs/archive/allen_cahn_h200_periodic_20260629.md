# Allen-Cahn H200 periodic-reencoding run, 2026-06-29

## Concrete results

Run `allen_cahn_multistable_pde_h200_periodic_20260629` completed 9 GPU jobs: LISTA KAE, dense KAE, and sparse-MLP KAE over seeds 0, 1, and 2. The benchmark used `allen_cahn_4`, grid size 16, two channels, state dimension 512, latent dimension 2048, tanh activations, no Koopman stability regularization, 10,000 updates, batch size 16, and training windows of length 20. The stored observation interval was `0.005 * 4 = 0.02`; H200 therefore corresponds to physical time 4.0. Dataset preflight generated three shared seed datasets and confirmed all four basins in train, validation, and test splits for each seed.

No-reencode mean field MSE over three seeds:

| Model | H20 | H50 | H100 | H200 |
|---|---:|---:|---:|---:|
| Dense KAE | 0.611153 | 0.920017 | 1.066243 | 1.173908 |
| LISTA KAE | 0.654826 | 0.930629 | 1.060638 | 1.113746 |
| Sparse-MLP KAE | 0.607846 | 0.900006 | 1.037929 | 1.147482 |

No-reencode terminal `final_field_mse`:

| Model | H20 | H50 | H100 | H200 |
|---|---:|---:|---:|---:|
| Dense KAE | 0.871477 | 1.189881 | 1.276971 | 1.363242 |
| LISTA KAE | 0.907691 | 1.142206 | 1.188703 | 1.142292 |
| Sparse-MLP KAE | 0.822235 | 1.146603 | 1.226334 | 1.371985 |

Paired no-reencode field-MSE differences versus dense were +0.043673, +0.010612, -0.005604, and -0.060162 for LISTA at H20, H50, H100, and H200 respectively. Sparse-MLP differences were -0.003307, -0.020011, -0.028314, and -0.026426.

## Context

The evaluator now records `forecast_modes` containing no-reencode plus periodic decode/re-encode periods 1, 5, 10, 20, and 50. Periodic reencoding uses only the model's own decoded predictions at refresh boundaries; it does not inject ground-truth future frames.

## Interpretation

The long no-reencode horizon is the clearest sparse forecasting signal so far in this Allen-Cahn setup. LISTA is worse than dense at H20 and roughly tied at H50/H100, but has the best H200 mean field MSE and much better H200 terminal error. Sparse-MLP is strongest around H50/H100 mean error but not H200 terminal error.

Periodic reencoding does not generally help here. Frequent refresh, especially period 1, worsens long-horizon error. Period 50 is the least damaging periodic mode and still shows LISTA better than dense at H100/H200, but the best H200 LISTA result remains no-reencode.

## Project implications

This run strengthens the case that long-horizon evaluation is necessary: the H20 pilot understated the distinction between models. It also suggests that the current learned decoder/encoder composition introduces drift when used repeatedly, so periodic reencoding is not a free stability fix for this PDE setting.

## Next steps

For paper-facing evidence, repeat with more seeds or a larger grid only if compute allows. Prioritize no-reencode H200/H300 evaluation and a code-density sweep over more periodic-refresh variants.
