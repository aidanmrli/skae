# SLURM launcher map

`scripts/` owns resource requests and dependency orchestration, not scientific
implementation. Python behavior is exposed through `skae-train`,
`skae-evaluate`, and `skae-paper`.

Launchers may select an execution subset, but they must not duplicate a frozen
system roster, model-row roster, or metric protocol as shell defaults. Task and
result manifests emitted by the Python workflow are the authoritative record;
queue manifests record operational facts such as job IDs, paths, and explicit
overrides.

Submit every file containing `#SBATCH` with `sbatch`. The maintained launchers
use the `long` partition. Do not execute their payloads on the login node.

## Layout

```text
common/
  cluster_env.sh                 portable scratch/cache resolution
  gpu_guard.sh                   CUDA preflight and utilization telemetry
  run_benchmark_task.sh          allocation-free single training payload
  run_benchmark_array.sh         one task per GPU allocation
  run_benchmark_packed_array.sh  several sequential tasks per GPU allocation
neurips_2026/
  controlled/                    controlled training, collection, alignment
  dysts/                         Dysts training, cache, evaluation, collection
  baselines/                     classical/local-linear controls
  allen_cahn_forecast_replication/ same-checkpoint new-IC robustness packet
  allen_cahn_periodic_reencoding_v5/ validation-frozen autonomous refresh test
  global_k_support_invariance/   held-out support-conditioned closure audit
  global_k_dense_zero_wd/        exact-dense specificity control and GPU gate
  global_k_distinct_laws_v2/     new-seed one-global-K distinct-law chain
  global_k_residual_forecast/    authorized residual one-global-K forecast chain
  local_operators/               staged routed-local-operator experiment
  interventions/                coordinate intervention replay
```

Submitted array wrappers call allocation-free payloads; packed workers never
execute another script containing `#SBATCH`.

## Controlled benchmark

- `neurips_2026/controlled/queue_training.sh`: build the frozen task table and
  submit packed training.
- `collect_forecasting.sh` and `compare_forecasting.sh`: collect reviewed runs
  and compare model roots.
- `queue_alignment.sh`: route all six paper rows to evaluation shards.
- `queue_alignment_shards.sh`, `reduce_alignment.sh`, and
  `merge_alignment.sh`: support-alignment dependency chain.

The reducer defaults to the canonical model rows in
`experiments/neurips_2026/alignment.py`. Pass `ROOT_LABELS_CSV` or
`ROOT_LABELS_FILE` only to run an explicit subset; the shell worker does not
carry another copy of the paper roster.

Canonical launch:

```bash
sbatch scripts/neurips_2026/controlled/queue_training.sh
```

## Dysts benchmark

- `neurips_2026/dysts/queue_training.sh`: frozen 10-system training campaign.
- `prebuild_cache.sh`: CPU cache materialization for explicit systems/profiles.
- `queue_evaluation.sh`: H100--H5000 evaluation dependency chain.
- `run_evaluation_array.sh`, `run_evaluation_packed_array.sh`, and
  `run_evaluation_task.sh`: submitted wrappers and allocation-free payload.
- `collect_evaluation.sh`: row collection after reevaluation.

Canonical launch:

```bash
sbatch scripts/neurips_2026/dysts/queue_training.sh
```

## Controls and mechanism experiments

- `neurips_2026/baselines/queue.sh`: standalone classical and mixture-local
  controls; `run_array.sh` executes task rows.
- `neurips_2026/local_operators/queue_training.sh`: staged support-routed local
  affine operator campaign; `run_array.sh` trains and `reevaluate.sh` applies
  the wide periodic grid.
- `neurips_2026/interventions/run.sh`: checkpoint-based coordinate intervention
  replay.
- `neurips_2026/global_k_support_invariance/`: CPU-only 45-checkpoint
  support-closure evaluation and system-level reduction. Both launchers require
  an explicit `OUTPUT_ROOT`; submit the reducer with an `afterok` dependency on
  the array. The evaluator emits both the persistent primary and an all-current
  no-next-state guard; the latter has a deterministic compact extractor under
  `experiments/neurips_2026/`.
- `neurips_2026/global_k_dense_zero_wd/`: prospective tanh/linear-latent,
  zero-weight-decay dense control. A packed utilization smoke is assessed
  without reading model outcomes; only a passing frozen utilization decision
  unlocks the three-GPU packed campaign, CPU evaluation array, and summary.
- `neurips_2026/allen_cahn_forecast_replication/`: cross the frozen 20 Allen--Cahn
  checkpoints with three newly generated datasets, reduce at the paired-checkpoint
  seed level, and authenticate the same-checkpoint new-IC robustness packet.
- `neurips_2026/allen_cahn_periodic_reencoding_v5/`: final outcome-blind
  execution chain for the frozen validation-selected periodic-reencoding test.
  Run its exact-shape A100L smoke first; independently pin the emitted smoke and
  CUDA-identity receipt hashes before submitting `queue.sh`, which chains the
  scientific GPU job to the guarded CPU summary. V1--V4 are invalid operational
  history and must not be used as scientific results.
- `neurips_2026/global_k_distinct_laws_v2/`: build and source-lock the paired
  sparse/dense seed-100--109 roster; gate the scientific run with an
  outcome-quarantined GPU smoke; then run checkpoint audit, physical-space
  mechanism evaluation, adjudication, packet authentication, and the frozen
  CPU-only supplemental finiteness/adjudication/per-basin audit.
- `neurips_2026/global_k_residual_forecast/`: frozen follow-up using one
  unchanged global `K` inside an autonomous, every-step-reencoded
  support-routed residual predictor. Its final V3 card/source/task/queue roots
  are `fdb48269a6a0f7f964fcbf27271f54a67f195f6ef46d2e5c83ebcf67046629ca`,
  `2c7439ca57c61e74c9f05b1dbb4d9f9c19c0e32efe60587063e27ae4ab8bd8e8`,
  `86a3dce2ce8fd6ca569aebcccb6812ac6c3ee206ec21ba8e2ccf2642305fb024`,
  and `db0222b88401214a34010e67ef0fdbf07d5d36d3ba9bc763249451a42afff8d4`.
  V3 is invalid at its frozen validity tier and directionally unadjudicated:
  five shards completed under quarantine, task 5 failed strict serialization
  on a nonfinite payload value (`inf`), four
  were not run, and no summary exists. V1/V2 operational failures and V3 are
  preserved in dated archive records; none is a scientific performance result,
  and no V4 is permitted.

Both mechanism launchers resolve scientific defaults from their Python
`protocol.py`/`contract.py` modules. Environment variables are explicit subset
or ablation overrides; shell files retain only resources, paths, dependency
order, and immutable checkpoint identities.

## Storage and portability

`common/cluster_env.sh` resolves storage in this order:

1. explicit `SKAE_SCRATCH_ROOT`;
2. `$SCRATCH/skae`;
3. `/network/scratch/<initial>/<user>/skae` when available;
4. repository-local `runs/`.

Set shared storage explicitly on another cluster:

```bash
export SKAE_SCRATCH_ROOT=/shared/path/skae
export DYSTS_CACHE_DIR=/shared/path/skae/dysts_native_cache
```

SLURM stdout/stderr paths are relative and contain no contributor name.
Launchers derive the repository root from Git, so their nested directory depth
does not affect path resolution.

A small set of completed, failed, or independently frozen prospective packets
retains byte-frozen contributor-specific targets. Their shell files remain at
their original paths solely so the recorded source manifests continue to
authenticate those exact bytes; they are provenance, not maintained launchers
to edit casually or reuse as templates. The
repository-architecture test enumerates those packets explicitly and pins each
manifest root, manifest entry count, affected script digest, and, for terminal
invalid experiments, the archived failure record. New or otherwise maintained
launchers do not inherit this exception and must resolve storage through
`common/cluster_env.sh`.

## Before a GPU submission

Read the launcher and called command, verify CUDA device selection, batching,
task packing, CPU/GPU stage separation, and telemetry. Use `sbatch` from the
repository root so relative log files and generated task manifests are easy to
find. The complete evidence workflow is documented in
[`experiments/neurips_2026/README.md`](../experiments/neurips_2026/README.md).
