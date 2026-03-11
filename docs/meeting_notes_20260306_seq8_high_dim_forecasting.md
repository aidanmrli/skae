# Meeting Notes: Seq8 Stability and High-Dimensional Forecasting

Date: March 6, 2026

## Goal

Primary goal: get good long-horizon forecasting MSE from a LISTA encoder.

Working metric:
- long-horizon forecasting, especially `H500` and `H1000` best-periodic MSE

Current framing:
- `sequence_length=8` (`seq8`) is the right default training mode for long-horizon forecasting
- the remaining question is whether the bottleneck is specifically LISTA, or whether the current Koopman autoencoder setup is itself a poor fit for some environments

## Main takeaways

- On `duffing`, longer-sequence training does improve long-horizon stability for LISTA.
- On `duffing`, training longer also matters: the `seq8` advantage is much clearer by `20000` steps than in shorter runs.
- Even after `seq8` tuning, LISTA still does not match the `generic_sparse` MLP encoder baseline on `duffing`.
- In intrinsic high dimensions, `competitive_lv` works, `kuramoto` is only partial, and `hopfield` is currently failing for all model families.
- If `generic_sparse` with a sufficiently large latent space also fails, then the issue is not just LISTA. It suggests the current Koopman autoencoder formulation may not be well suited for long-horizon forecasting on that environment.

## Seq8 result on Duffing

### 1. `L=1` vs `L=8` at `10000` steps

Unstructured LISTA on `duffing`, `3` seeds:

| setting | quick eval best | `H100` best-periodic | `H500` best-periodic | `H1000` best-periodic |
|---|---:|---:|---:|---:|
| `L=1` | `1.0058` | `2.227e-02` | `2.9644` | `5.616e+04` |
| `L=8` | `1.1251` | `1.554e-02` | `3.971e-01` | `8.172e-01` |

Interpretation:
- `L=8` is slightly worse on quick eval.
- `L=8` is dramatically better on long-horizon forecasting.
- `L=1` produced a catastrophic long-horizon outlier; `L=8` was much more robust.

### 2. `L=1` vs `L=8` at `20000` steps

Same setup, but trained longer:

| setting | quick eval best | `H100` best-periodic | `H500` best-periodic | `H1000` best-periodic |
|---|---:|---:|---:|---:|
| `L=1` | `0.6468` | `4.344e-03` | `1.115e-01` | `4.825e-01` |
| `L=8` | `0.7194` | `1.288e-03` | `4.890e-02` | `2.220e-01` |

Interpretation:
- both modes improve with more training
- `L=8` remains the better long-horizon choice
- the remaining tradeoff is that `L=1` is still a bit better on quick/short-horizon metrics

Conclusion from Duffing:
- `seq8` improves stability and long-horizon forecasting for LISTA
- longer training is important; `>= 20000` steps is a better basis for comparison than `10000`
- this solves part of the stability problem, but not the LISTA-vs-MLP performance gap

## Current best LISTA Seq8 status on Duffing

ReLU-final LISTA tuning at `50000` steps improved the `seq8` baseline further.

Best in-band LISTA anchor so far:
- `sp_0p0060_loops_1`
- quick `0.3162`
- `H500=0.0119`
- `H1000=0.0919`
- sparsity `0.8490`

Reference `generic_sparse` anchor:
- quick `0.1115`
- `H500=0.00309`
- `H1000=0.0294`

Interpretation:
- LISTA can be made reasonably stable on `duffing`
- but the best tuned LISTA arm is still worse than `generic_sparse` by about `2.84x` on quick eval, `3.85x` on `H500`, and `3.12x` on `H1000`
- so `seq8` helps, but it does not close the encoder gap by itself

## Intrinsic high-dimensional experiments

Shared baseline:
- `sequence_length=8`
- `target_size=256`
- `num_steps=10000`
- current defaults: `kuramoto=16`, `hopfield=16`, `competitive_lv=10`

Seed-median `H1000` best-periodic MSE:

| system | `generic_sparse` | `lista_blockdiag` | `lista_dense` | status |
|---|---:|---:|---:|---|
| `competitive_lv` | `0.0651` | `0.1192` | `0.1654` | solved positive control |
| `kuramoto` | `199.86` | `258.50` | `6.636e8` | partial success, robustness problem |
| `hopfield` | `5436.67` | `3.599e15` | `3.045e33` | current blocker |

### System-by-system readout

`competitive_lv`
- all three model families are stable enough to forecast well at `H1000`
- this validates that the intrinsic-HD benchmark setup is not broken
- it is the positive control

`kuramoto`
- `generic_sparse` is best
- `lista_blockdiag` is second-best but has a catastrophic seed
- dense LISTA is not viable
- this looks like a recoverable tuning/stability problem, but not a solved forecasting result

`hopfield`
- all model families are poor at long horizon
- `generic_sparse` is least bad, but still far from acceptable if the target is good long-horizon MSE
- LISTA variants are catastrophically unstable
- this is currently the strongest evidence that the issue may be larger than encoder choice alone

## Important hypothesis for discussion

We should explicitly separate two cases:

### Case 1: `generic_sparse` works, LISTA does not

Interpretation:
- the environment is forecastable with the current Koopman autoencoder family
- the problem is likely LISTA-specific: encoder capacity, thresholding, optimization, or support quality

### Case 2: `generic_sparse` also fails

Interpretation:
- the problem is probably not just LISTA
- the current Koopman autoencoder setup may simply not be well suited for long-horizon forecasting on that environment
- in that regime, continuing to tune LISTA alone is unlikely to solve the core issue

Current evidence:
- `competitive_lv` suggests the Koopman AE approach can work in intrinsic high dimension
- `kuramoto` is ambiguous: the model family may still be recoverable, but current MSE is not yet good
- `hopfield` is the clearest warning case, because even the MLP encoder baseline is poor

## Status of follow-up high-dimensional runs

- The baseline intrinsic-HD sweep is complete and has usable results.
- A dedicated `kuramoto` recovery follow-up has been scaffolded, but the current collected summary is still empty, so there is no new consolidated result from that recovery path yet.
- The stricter intrinsic-HD sizes (`N=32/64`) have not been run yet.

## Recommended decisions

1. Keep `seq8` as the default training mode for long-horizon forecasting experiments.
2. Keep `generic_sparse` as the control model for judging whether an environment is forecastable under the current Koopman AE setup.
3. For intrinsic-HD follow-up, carry forward `generic_sparse` and `lista_blockdiag`; drop dense LISTA unless there is a specific ablation reason.
4. Treat `hopfield` primarily as a Koopman-family suitability / stability question, not only a LISTA tuning question.
5. Do not move to stricter `N=32/64` runs until the current `kuramoto` recovery question is answered.

## Immediate next steps

1. Finish the dedicated `kuramoto` recovery pass and check whether `lista_blockdiag` can get close to the `generic_sparse` control under better `seq8` settings.
2. For `hopfield`, test whether any `generic_sparse` setting with enough capacity and training budget can produce acceptable `H1000` MSE before spending more time on LISTA-specific tuning.
3. If `generic_sparse` still fails on `hopfield`, consider whether a different representation or stronger stability control is needed rather than more encoder tuning.
4. Use the following decision rule going forward:
   - if `generic_sparse` is good and LISTA is bad -> keep working on LISTA
   - if both are bad -> question the Koopman forecasting formulation for that environment
