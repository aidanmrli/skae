# Paper launchers

`scripts/` contains only the launch and orchestration surface for evidence that
is still represented in `docs/neurips_sparse_koopman_multibasin.tex`. Python
implementations are mapped in `tools/README.md`. Submit scripts with `#SBATCH`
headers using `sbatch`; do not execute their training or evaluation payloads on
the login node.

## Shared workers

- `run_paper_benchmark_array.sh` and
  `run_paper_benchmark_packed_array.sh`: submitted controlled/Dysts training
  wrappers. Both invoke the allocation-free `run_paper_benchmark_task.sh`
  payload; packed workers never execute a script containing `#SBATCH`.
- `slurm_gpu_guard.sh`: GPU-use preflight sourced by training workers.
- `collect_transition_rich_basin_partition.sh` and
  `compare_paper_benchmark.sh`: controlled collection and comparison.

## Controlled multibasin evidence

- `queue_controlled_paper_training.sh`: canonical 15-system, six-row, 15-seed
  training queue. Optional system, model, and seed subsets support targeted
  repairs without restoring historical one-off launchers. It packs tasks
  sequentially on one GPU through the shared benchmark worker.
- `queue_transition_rich_interpretability_shards.sh`,
  `reduce_transition_rich_interpretability_metrics.sh`, and
  `merge_transition_rich_interpretability_shards.sh`: current support-alignment
  evaluation chain.
- `queue_controlled_support_alignment.sh`: fixed absolute-1e-3,
  Jaccard-0.50 alignment evaluation for all controlled rows. Families are fit
  on every generated evaluation-trajectory state, then scored on each observed
  label's tie-inclusive center-margin-at-or-above-q75 slice.
- Per-system controlled tables are generated from frozen evidence by
  `tools/build_controlled_per_system_tables.py`.

## Dysts evidence

- `queue_dysts_dt30_basinblock_p256_seeds0to14.sh`: final Dysts training queue.
- `prebuild_dysts_cache_matrix.sh`: deterministic CPU cache worker. Callers
  must provide a generated system file and explicit SLURM array range; paper
  defaults are the retained 10 systems, the full cache profile, and
  (dt\times30).
- `queue_dysts_long_horizon_eval.sh`,
  `run_dysts_long_horizon_eval_array.sh`, and
  `run_dysts_long_horizon_eval_packed_array.sh`: H100--H5000 paper evaluation
  chain. The queue requires an explicit root-spec table from training. Both
  submitted wrappers invoke `run_dysts_long_horizon_eval_task.sh`, which has
  no allocation directives.
- `collect_dysts_long_horizon_forecasting.sh`: Dysts collector.

## Baselines, staged models, and interventions

- `queue_paper_baseline_suite.sh` and `run_paper_baseline_suite.sh`: retained
  classical and local-linear baselines.
- `queue_staged_support_family_local_k_table1.sh` and
  `run_staged_support_family_local_k_array.sh`: current staged F_abs
  local-operator training. The protocol is source-versioned: 512 configured
  route-fit rows contain two exact copies of 256 unique trajectories, and the
  fixed staged selector uses 32 starts at seed offset 12345. Support routing is
  recomputed before every latent transition, while the selected periodic cadence
  controls decode--encode refresh. Resumable `last.pt` checkpoints are enabled
  by default; schema-3 checkpoints include local optimizer and stochastic-stream
  states.
- `reevaluate_staged_vs_global_wide_periodic.sh`: staged/global forecasting
  reevaluation on the 100-start wide cadence grid; it does not repair the
  staged selector's 32/100 overlap or the global selector asymmetry.
- `run_support_coordinate_interventions.sh`: reproduce the intervention run;
  frozen rows and paper displays are verified by the builders in
  `tools/README.md`.

Historical sweeps, rescue queues, galleries, and standalone mechanism campaigns
are intentionally absent. Use Git history for their implementation provenance;
do not recreate them as active launchers without first changing the paper plan.
