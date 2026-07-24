# Corrected Dysts dt×30 v4 smoke-gate failure — 2026-07-22

## Concrete result

GPU smoke job `10173592_0` completed all 12 requested Chua fits in 4:59. The trace contained 10 post-startup samples with 99.11% mean and active GPU utilization and 100% peak utilization. All 12 training receipts used the corrected schema-3 cache, direct held-out checkpoint selection, and the requested recipes. Gate job `10173593` nevertheless failed, so full-cache, full-training, and evaluation jobs did not run.

## Experimental context

The failed gate treated the unused default `ENCODER_TYPE="lista"` field in dense and sparse-MLP `GenericKM` configurations as evidence that they were LISTA models. It therefore demanded one LISTA refinement from the dense tanh control. A second replay-only issue was that the gate attempted to cast descriptive metric metadata to a float.

## Interpretation

This was an adjudicator false negative, not a failed model, cache, provenance, or utilization smoke. It produced no forecasting comparison and is not scientific evidence.

## Project implications

The gate now scopes the one-refinement assertion to actual `LISTAKM` models, requires finite scalar endpoints, and permits descriptive metadata. Regression tests cover both failure modes. Replaying the repaired gate over the untouched v4 artifacts passes all checks.

## Next step

Use only the fresh v5 dependency chain and preserve v4 unchanged as operational provenance.
