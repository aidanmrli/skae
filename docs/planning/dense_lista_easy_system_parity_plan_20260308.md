# Dense LISTA Easy-System Parity Plan

Date: 2026-03-08

## Goal

- Improve dense LISTA on the easy cross-system near-miss cases without changing model architecture.
- Keep the architecture fixed across systems for fairness.
- Only tune external knobs: training length, optimizer learning rates, and later loss coefficients / checkpoint selection.

## Current Status

- Stage 1 is complete under `results/dense_lista_easy_parity_stage1_20260308`.
- Best win-count recipe: `lista_dense_ns100k_lr5em5_klr5em6_wd1em4` wins `6/8` target systems with median dense/generic ratio `0.8699`.
- Best median-ratio recipe: `lista_dense_ns200k_lr5em5_klr5em6_wd1em4` wins `5/8` target systems with median ratio `0.7888`.
- Remaining holdouts after Stage 1:
  - `competitive_lv`
  - `duffing`
- Stage 2 is now queued under `results/dense_lista_easy_parity_stage2_20260308` as a coefficient-only holdout sweep on those two systems.

## Numerical Basis

On the canonical `v4` paper benchmark, dense LISTA is already close to `generic_sparse` on the cross-system `H1000` best-periodic metric:

- `generic_sparse`: `0.0328`
- `lista_dense`: `0.0388`
- dense LISTA wins `15/29` systems and loses `14/29`
- paired median dense/generic ratio: `0.9588`

That is near parity overall, but several accepted-default systems are still clear near-misses where `generic_sparse` stays better.

## Target Systems

These are the easy dense-LISTA losses to target first. They are all accepted-default systems in the `v4` benchmark, so we can keep the benchmark-selected `dt` fixed and focus on optimization.

| system | dense / generic `H1000` ratio |
|---|---:|
| `dysts:SanUmSrisuchinwong` | `1.006` |
| `dysts:Hadley` | `1.415` |
| `multiwell_gradient` | `1.493` |
| `dysts:LuChenCheng` | `1.582` |
| `duffing` | `2.067` |
| `dysts:Dadras` | `2.069` |
| `blended` | `2.078` |
| `competitive_lv` | `2.841` |

## Fairness Constraints

- Freeze architecture completely for this campaign.
- Keep the same dense LISTA architecture on every system:
  - config: `lista_parity_generic_sparse`
  - `target_size=256`
  - `sequence_length=8`
  - dense Koopman `K`
  - `lista_alpha=0.15`
  - `lista_num_loops=1`
  - `lista_final_op=relu`
- Keep the benchmark-selected `dt` fixed for these easy systems.
- Do not use extra `dt` halving in this parity campaign.
- Compare every new dense-LISTA arm against the fixed `generic_sparse` `v4` anchor.

## Stage 1

Run a dense-LISTA optimization sweep on the 8 target systems with coefficients fixed to the benchmark recipe:

- `reconst_coeff=0.03`
- `pred_coeff=1.0`
- `sparsity_coeff=0.006`
- `res_coeff=1.0`
- `weight_decay=1e-4`

Sweep:

- `num_steps in {50000, 100000, 200000}`
- `(lr, k_matrix_lr) in {(1e-4, 1e-5), (3e-4, 3e-5), (5e-5, 5e-6)}`
- seeds `{0,1,2}`

This yields `8 systems x 3 step lengths x 3 lr pairs x 3 seeds = 216` training runs.

## Decision Rule After Stage 1

- Promote the best dense-LISTA external recipe if it improves the 8-system target-set median dense/generic ratio below `0.9` or flips several of the current near-miss systems.
- Reject recipes that improve a few systems only by introducing new catastrophic failures.
- If multiple arms are close, prefer the simpler story:
  - same benchmark `dt`
  - same architecture
  - same coefficients
  - only optimization changed

## Stage 2

After Stage 1, hold the winning optimizer/length setting fixed and sweep coefficients one axis at a time:

- `sparsity_coeff in {0.003, 0.006, 0.012}`
- `reconst_coeff in {0.01, 0.03, 0.1}`
- `pred_coeff in {0.5, 1.0, 2.0}`

Do not run the full cross-product unless the first axis sweeps show clear gains.

### Active Stage-2 Matrix

- Systems:
  - `competitive_lv`
  - `duffing`
- Base recipes:
  - `100k`, `lr=5e-5`, `k_matrix_lr=5e-6`, `weight_decay=1e-4`
  - `200k`, `lr=5e-5`, `k_matrix_lr=5e-6`, `weight_decay=1e-4`
- Coefficient variants per base recipe:
  - baseline
  - `sparsity_coeff in {0.003, 0.012}`
  - `reconst_coeff in {0.01, 0.1}`
  - `pred_coeff in {0.5, 2.0}`
- Seeds:
  - `{0,1,2}`
- Total jobs:
  - `84`

## Stage 3

For the top few Stage-1/2 recipes, compare checkpoint-selection rules:

- current validation-selected checkpoint
- long-horizon paper-metric checkpoint
- last-checkpoint diagnostic

This stays architecture-fixed and tests whether selection mismatch is part of the remaining dense-LISTA gap.

## Promotion Path

1. Run Stage 1 on the 8 easy systems.
2. Pick one winning external recipe.
3. Rerun that single recipe on the full 29-system benchmark.
4. Only after that, transfer the same recipe to `lista_blockdiag` as a secondary readout.

## Interpretation

- Longer training is a fair first lever for the easy systems.
- Learning-rate tuning is a fair first lever for the easy systems.
- Extra `dt` reduction is not part of this main parity campaign because it changes task difficulty at fixed-step horizons.
- Hard intrinsic-HD systems (`kuramoto`, `hopfield`) remain a separate track.
