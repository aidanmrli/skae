# Dysts dt×30 cadence audit — 2026-07-22

## Concrete result

The historical KAE Dysts caches did not realize the paper-described observation interval. The wrapper recorded a `dt` override, but the native Dysts trajectory generator continued to integrate at each system's intrinsic `dt`. Historical KAE horizons H100/H2000/H4000/H5000 therefore covered approximately 0.02/0.40/0.80/1.00 characteristic periods, not 0.60/12/24/30. Standalone controls used the intended `30dt` cadence, so historical cross-model comparisons are not matched.

A repaired numerical validator now passes for all ten systems. The installed timestep and returned observation increments realize a multiplier of 30, and nine corrected observations agree with every thirtieth native-grid observation to normalized maximum differences between approximately (4\times10^{-15}) and (5\times10^{-13}).

## Experimental context

The correction propagates the override into the native integrator, bumps the cache schema to version 3, validates exact metadata/shape/count/dtype/finiteness, and creates split-specific perturbed initial conditions for train, validation, policy, and test. All six KAE rows must be retrained; LISTA, LISTA-BD, and LISTA-SB use one learned refinement, and Dense MLP uses tanh with no sparsity source.

Two replacement launch attempts were stopped before valid training evidence. The first preflight rejected an ordering mismatch. The second exposed two launcher defects: the policy split was missing from a CLI allowlist, and GPU workers inherited the legacy cache root instead of the isolated corrected root. The invalid GPU smoke was cancelled after 2:26 with 0% utilization samples while workers redundantly attempted cache construction. Its corrected smoke train/validation cache files remain valid.

## Interpretation

No historical Dysts forecasting value, significance annotation, ratio, or horizon curve supports the paper. This is a provenance/cadence failure, not a negative result for sparse KAEs or for one-refinement LISTA-SB. The corrected campaign is a new discrete-time forecasting problem and must not reuse native-dt checkpoints.

## Project implications

The active manuscript now excludes historical Dysts outcomes and retains only a transparent quarantine statement. Corrected checkpoint selection and headline evaluation both use direct repeated-(K) rollout. Evaluation tracks each initial condition independently, ranks optional periodic modes by full-horizon survival before strict MSE, and the collector requires strict direct full-horizon metrics. System-level analysis pairs seeds within system before treating the ten systems as inferential units.

## Next steps

Run the versioned 12-fit corrected GPU smoke with explicit cache-path validation and a utilization gate. Only if it passes, build all four full cache splits, train all 900 matched fits, evaluate strict direct horizons through H5000, and adjudicate the complete system-level packet. Periodic reencoding remains a secondary deployment sensitivity; test-oracle best-periodic values cannot support the headline.
