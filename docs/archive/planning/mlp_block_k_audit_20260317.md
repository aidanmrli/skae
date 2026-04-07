# MLP Block-Diagonal `K` Audit

Date: March 17, 2026
Status: done

## Summary

The historical `generic_sparse + block_diagonal K` control runs are not valid block-diagonal-`K` MLP experiments.

The launch scripts and saved configs requested `K_STRUCTURE=block_diagonal`, but the `GenericKM` implementation in [skae/model.py](/home/mila/l/lia/skae/skae/model.py#L890) does not branch on `cfg.MODEL.K_STRUCTURE`. It always instantiates a dense latent transition matrix `self.kmat` and always returns that dense matrix in `kmatrix()`. As a result, the paper-facing `Sparse MLP + block-K` rows are mislabeled dense-`K` MLP reruns, not valid structure-isolation controls.

This invalidates the historical MLP `+ block-K` rows in:
- corrected 4-basin `competitive_lv`
- Kuramoto `N=16`
- Hopfield `N=64` quarter-step
- higher-basin fixed-system `competitive_lv`
- the block-`K` fairness-control summary table

The affected runs should be kept only as provenance for the failed control attempt. They should not be used for causal claims about block-diagonal latent transitions until the MLP implementation is fixed and the experiments are rerun.

## What Was Expected

The intended fairness-control design was:
- keep the sparse MLP encoder/decoder stack fixed
- change only the latent transition from dense to block-diagonal
- then compare against dense-`K` sparse MLP and block-diagonal LISTA

The queue scripts do request exactly that setup. For example:
- [scripts/queue_clv_15seed_extension.sh](/home/mila/l/lia/skae/scripts/queue_clv_15seed_extension.sh#L68) queues `generic_sparse_blockdiag_200k` with `block_diagonal 16`
- [scripts/run_clv_15seed_extension_array.sh](/home/mila/l/lia/skae/scripts/run_clv_15seed_extension_array.sh#L60) forwards `--k_structure` and `--k_block_size`

The saved run config for an affected CLV control run also records the intended setting:
- [config.json](/network/scratch/l/lia/skae/clv_15seed_extension_20260311/generic_sparse_blockdiag_200k/competitive_lv/dt_0p01/seed_10/20260310-223639/config.json#L100) has `"MODEL"."K_STRUCTURE": "block_diagonal"` and `"K_BLOCK_SIZE": 16`

So the failure is not that the launch scripts forgot to request the block-diagonal structure. The failure is downstream of that request.

## Root Cause

`GenericKM` ignores `cfg.MODEL.K_STRUCTURE`.

In [skae/model.py](/home/mila/l/lia/skae/skae/model.py#L924), `GenericKM` always creates

```python
self.kmat = nn.Parameter(torch.eye(cfg.MODEL.TARGET_SIZE))
```

and in [skae/model.py](/home/mila/l/lia/skae/skae/model.py#L968) it always returns

```python
return self.kmat
```

There is no `dense` / `diagonal` / `block_diagonal` switch in `GenericKM`.

By contrast, `LISTAKM` does implement structure-aware latent dynamics beginning at [skae/model.py](/home/mila/l/lia/skae/skae/model.py#L1036). That is why the bug is specific to the MLP `+ block-K` control and does not by itself invalidate the block-diagonal LISTA runs.

## Direct Checkpoint-Level Confirmation

I inspected a saved checkpoint from the supposed block-diagonal MLP CLV control:
- `/network/scratch/l/lia/skae/clv_15seed_extension_20260311/generic_sparse_blockdiag_200k/competitive_lv/dt_0p01/seed_10/20260310-223639/checkpoint.pt`

The checkpoint state dict contains:
- dense `kmat`
- no block-diagonal-specific latent-transition parameters

The saved `kmat` is `256 x 256`, and its off-block entries are not numerically negligible:
- off-block absolute sum is about `69.99`
- off-block entries above `1e-8`: `61,438`

So the run is not merely "recorded ambiguously"; the learned operator itself is dense.

## Why the Results Look Suspicious

The corrected CLV `15`-seed collector shows many seed-level rows that are numerically identical between:
- `generic_sparse_ns200k_best`
- `generic_sparse_blockdiag_200k`

For example, the `H3000` system-median tie in [forecasting_summary.md](/home/mila/l/lia/skae/results/clv_15seed_extension_20260311/collect_200k_15seed_full/forecasting_summary.md#L69) is real as an order statistic, but it is not evidence that a valid block-diagonal control matched the dense baseline. It comes from comparing two dense-`K` MLP runs that differ only by a requested setting that `GenericKM` ignored.

This also explains why several seed-level best-periodic values match exactly across the two roots in [forecasting_rows.csv](/home/mila/l/lia/skae/results/clv_15seed_extension_20260311/collect_200k_15seed_full/forecasting_rows.csv#L17): same architecture family, same seed, same effective latent-dynamics class.

## Scope of Invalidated Artifacts

The issue affects every historical `Sparse MLP + block-K` row that relied on `GenericKM` with a requested block-diagonal latent transition. In the current paper packet this includes:
- Kuramoto `N=16`, identical frequencies
- Hopfield `N=64`, quarter-step setting
- corrected 4-basin `competitive_lv` full `15`-seed table
- higher-basin fixed-system `competitive_lv`
- the block-`K` fairness-control summary table

It also invalidates the earlier interpretation that these controls "closed the pure-`K` fairness question negatively." That conclusion depended on MLP controls that were not actually structured-`K` runs.

## Interpretation

This is a model-implementation bug with paper-facing consequences.

What it does **not** imply:
- it does not by itself invalidate dense-`K` sparse MLP results
- it does not by itself invalidate block-diagonal LISTA runs
- it does not prove that a valid MLP block-diagonal `K` control would help

What it **does** imply:
- the existing MLP `+ block-K` rows are not valid evidence
- any claim that "block-diagonal `K` alone does not explain the result" is currently unsupported by these MLP controls
- the affected rows must be marked invalid in the docs and rerun after a code fix

## Required Next Steps

1. Implement actual `K_STRUCTURE` support for `GenericKM`, or create a separate MLP model class that supports dense / diagonal / block-diagonal latent transitions.
2. Add a direct unit or checkpoint-level test that fails if a requested block-diagonal MLP run produces a dense off-block `kmat`.
3. Rerun the affected MLP `+ block-K` fairness controls.
4. Replace or unmark the current historical rows in the paper docs only after the rerun is collected.

## Immediate Documentation Policy

Until reruns exist:
- keep the historical numerical rows only as invalidated artifacts
- mark every `Sparse MLP + block-K` row in the review tables as invalid
- note in paper-status docs that the fairness-control conclusion is reopened
- do not use these rows to support causal or mechanistic claims
