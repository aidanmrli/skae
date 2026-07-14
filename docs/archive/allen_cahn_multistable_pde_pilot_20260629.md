# Allen-Cahn multistable PDE pilot, 2026-06-29

## Concrete results

The Allen-Cahn pilot `allen_cahn_multistable_pde_pilot_20260628` completed 9 GPU runs: LISTA KAE, dense KAE, and sparse-MLP KAE across seeds 0, 1, and 2. The launch manifest used `allen_cahn_4`, grid size 16, two channels, state dimension 512, latent dimension 2048, tanh activations, and no Koopman stability regularization. Dataset preflight confirmed that all four basin labels were represented in train, validation, and test splits for each seed.

Mean field MSE over three seeds was:

| Model | H1 | H4 | H8 | H12 | H20 |
|---|---:|---:|---:|---:|---:|
| Dense KAE | 0.272808 | 0.271640 | 0.287532 | 0.346199 | 0.516179 |
| LISTA KAE | 0.272040 | 0.277316 | 0.293157 | 0.344198 | 0.511920 |
| Sparse-MLP KAE | 0.272537 | 0.271983 | 0.289638 | 0.351335 | 0.517975 |

Paired mean field-MSE differences versus dense were -0.000768, 0.005676, 0.005625, -0.002001, and -0.004259 for LISTA at horizons 1, 4, 8, 12, and 20 respectively. Negative values favor LISTA. Sparse-MLP differences versus dense were -0.000271, 0.000343, 0.002106, 0.005135, and 0.001795.

## Context

This pilot was launched after enforcing the spatialized PDE overcomplete-latent rule, so `d_z = 4 d_x`. Dense and sparse model families used the same convolutional tanh front end. The evaluation recorded field MSE, final-field MSE, gradient MSE, Fourier-band errors, basin-pixel metrics, and basin-consistency metrics over horizons 1, 4, 8, 12, and 20.

## Interpretation

The pilot shows that LISTA is competitive and has the lowest mean field MSE at horizons 1, 12, and 20, including the longest evaluated horizon. The effect is small relative to three-seed variation, and dense KAE is better at intermediate horizons 4 and 8. These results support continuing with Allen-Cahn as a high-dimensional multibasin benchmark, but they do not yet support a strong paper claim that sparse LISTA reliably outperforms dense KAE.

## Project implications

The setup satisfies the core protocol constraints for a high-dimensional multibasin PDE: overcomplete latent, tanh dense baseline, no K regularization, equal training budget, all basins represented before training, and horizon-wise forecasting evaluation. The result direction is encouraging for long-horizon LISTA forecasting but needs a larger grid or longer rollout window, more updates, and possibly a sparse-code hyperparameter sweep before it should affect the main paper narrative.

## Next steps

Use the generated artifacts in `results/allen_cahn_multistable_pde_pilot_20260628/` as the pilot evidence package. The next paper-relevant run should increase resolution or sequence length and sweep LISTA sparsity/code-density while keeping the same dense tanh baseline and the same overcomplete latent ratio.
