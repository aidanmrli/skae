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

## Before a GPU submission

Read the launcher and called command, verify CUDA device selection, batching,
task packing, CPU/GPU stage separation, and telemetry. Use `sbatch` from the
repository root so relative log files and generated task manifests are easy to
find. The complete evidence workflow is documented in
[`experiments/neurips_2026/README.md`](../experiments/neurips_2026/README.md).
