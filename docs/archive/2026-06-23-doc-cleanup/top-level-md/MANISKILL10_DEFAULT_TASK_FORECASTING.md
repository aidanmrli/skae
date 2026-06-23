# ManiSkill-10 Default-Task Forecasting Benchmark

## Purpose

This benchmark is a forecasting-generalization benchmark, not the primary
support-alignment benchmark. It asks whether the learned controlled Koopman
representation is useful across a diverse set of official ManiSkill state/action
demonstration tasks when evaluated with decoded-prediction periodic
re-encoding. It should be reported separately from the perturbation-balanced
insertion benchmark, whose role is outcome/contact-regime discovery.

Training remains label-free. The trainer consumes only compact observations,
actions, valid masks, and train/validation/test splits. Task success flags and
other labels, if present in the downloaded trajectory files, are optional
evaluation metadata and must not be used for checkpoint selection, model
training, support construction, or periodic re-encoding.

## Task Roster

The primary roster is stored in
`experiments/maniskill10_default_tasks.tsv`.

| Task ID | Max episode steps | Primary horizons | Notes |
| --- | ---: | --- | --- |
| `PickCube-v1` | 50 | `10,20,30,40,50` | single-object pick/place |
| `PushCube-v1` | 50 | `10,20,30,40,50` | planar pushing |
| `PullCube-v1` | 50 | `10,20,30,40,50` | planar pulling |
| `PokeCube-v1` | 50 | `10,20,30,40,50` | peg-mediated contact |
| `StackCube-v1` | 50 | `10,20,30,40,50` | two-object stacking |
| `RollBall-v1` | 80 | `10,20,40,60,80` | rolling contact |
| `PullCubeTool-v1` | 100 | `10,20,50,75,100` | tool-use pulling |
| `PushT-v1` | 100 | `10,20,50,75,100` | precision planar pushing |
| `PegInsertionSide-v1` | 100 | `10,20,50,75,100` | default-demo insertion only |
| `PlugCharger-v1` | 200 | `10,20,50,100,150,200` | harder insertion, longer default horizon |

`LiftPegUpright-v1` is the first replacement task if `PlugCharger-v1` or
another row fails replay/compaction on the cluster. It is also a useful
one-task smoke test because its default horizon is only 50 steps.

## Data Preparation Contract

Prepare one compact NPZ per task:

`data/maniskill/default_tasks/<TASK_ID>_state_compact_seed0.npz`

Use the existing compact format from
`skae.benchmarks.maniskill_insertion_dataset`: `observations`, `actions`,
`valid`, trajectory-level `split`, optional `outcome`, optional
`contact_phase`, `feature_names`, `action_names`, and `metadata_json`.

The current compactor is already mostly task-agnostic despite the insertion
name. It flattens numeric leaves from `obs` or `env_states`, appends previous
actions by default, and derives train/val/test splits at the trajectory level.
For the default-task benchmark:

1. Download demos with `uv run --with mani_skill --with h5py python -m mani_skill.utils.download_demo <TASK_ID>`.
2. Prefer downloaded HDF5 files that already contain `actions` and
   `env_states`. Replay is only needed if a selected HDF5 lacks usable state
   arrays or if we decide to standardize on `obs_mode=state`.
3. Use each task's default max episode steps as the compaction cap for the
   primary benchmark. This keeps `H=max_episode_steps` comparable to ManiSkill's
   task table rather than using the old insertion default of `MAX_STEPS=150`.
4. Keep every task as a separate dataset and train separate models per task.
   Do not concatenate tasks into a shared observation space unless a later
   multi-task model explicitly standardizes feature names and action spaces.

## Model Grid

Use the same hidden width/depth across rows:

| Row | Encoder | Activation | Sparsity | LISTA alpha | Optimizer |
| --- | --- | --- | ---: | ---: | --- |
| dense baseline | dense MLP | tanh | `0` | n/a | include standard and tuned fairness settings |
| sparse MLP | dense MLP | ReLU | `0.003` | n/a | matched to dense optimizer |
| sparse MLP | dense MLP | ReLU | `0.01` | n/a | matched to dense optimizer |
| LISTA | LISTA | ReLU | `1e-4` | `0.01` | standard optimizer first |

The dense no-sparsity baseline must remain tanh. If a dense ReLU ablation is
run, label it as an ablation, not as the dense baseline.

Set the latent dimension after data preparation from the observed state
dimension, not as a blind constant:

`z_dim = max(320, 64 * ceil((4 * obs_dim) / 64))`

This preserves an overcomplete Koopman lift even when a task has a larger state
vector than `PegInsertionSide-v1`.

## Evaluation

Primary metric:

- Mean state MSE on the held-out test split, aggregated by task and horizon,
  using the best score over fixed periodic re-encoding periods.

Periodic re-encoding grid:

`1,2,5,10,20,50,100`

For each task, only report horizons with full or explicitly documented held-out
coverage. The headline endpoint should be the task's default max episode steps
when enough held-out episodes have that many transitions. Reduced-coverage
longer rows are diagnostics, not benchmark headline claims.

Recommended aggregate displays:

1. Per-task curve of best-periodic test MSE versus horizon.
2. Cross-task geometric mean ratio versus dense tanh at each normalized horizon
   bucket.
3. Win counts by task at each task endpoint.
4. Optional no-reencoding appendix curve to show how much periodic refresh
   stabilizes each model.

Support-alignment metrics are not the main claim for this default-task suite.
Most default demonstration packets are mostly successful expert-like behavior,
so support labels may be low-entropy or absent. Outcome/contact support reads
can be included as exploratory diagnostics only when labels exist and have
enough held-out diversity.

## Code Generalization Needed

Small safe pieces:

- Add a general prepare wrapper that reads
  `experiments/maniskill10_default_tasks.tsv`, downloads each task, locates the
  best HDF5 candidate, and calls `tools/maniskill_prepare_insertion_dataset.py`
  with the task-specific `max_episode_steps`.
- Add a packed GPU training launcher that iterates over
  task/model/seed/checkpoint rows and uses the existing
  `tools/train_maniskill_controlled_lista.py`.
- Add a CPU evaluation launcher that evaluates saved checkpoints with
  `tools/evaluate_maniskill_controlled_lista.py` and task-specific horizons.
- Add a summarizer that aggregates `metrics_summary.json` files into per-task,
  per-horizon ratios against dense tanh.

Larger cleanup, not required before the first queue:

- Rename insertion-specific module/script names to generic ManiSkill controlled
  forecasting names. This is cosmetic for the benchmark but should wait until
  Worker A's perturbation-balanced insertion path is stable.
- Improve feature-group metrics beyond peg/hole-specific groups. The current
  `feature_group_indices` is insertion-biased; default-task forecasting can use
  whole-state MSE first.

## Queue Plan

Stage 0, CPU data smoke:

- One task: `LiftPegUpright-v1` or `PickCube-v1`
- Resources: `long`, `4` CPUs, `16G`, no GPU, `1` hour
- Output: compact NPZ and summary JSON; verify action/state dimensions and
  horizon coverage.

Stage 1, CPU data preparation:

- Ten-task array over the manifest
- Resources: `long`, `4` CPUs, `16G`, no GPU, `4` hours
- Output: `10/10` compact NPZ files and summaries

Stage 2, packed GPU training smoke:

- Two tasks, one seed, four model rows, `5000` steps
- Resources: `long`, one GPU, `8` CPUs, `24G`, `<=3` hours
- Pack four concurrent small KAE workers per GPU and log `nvidia-smi`
  telemetry. Refuse launches with `PACK_CONCURRENCY < 2` unless batch size is
  increased enough to justify a single process.

Stage 3, CPU evaluation smoke:

- Evaluate the Stage 2 checkpoints on CPU with the periodic grid
- Resources: `long`, `4` CPUs, `16G`, no GPU, `<=3` hours

Stage 4, paper-facing pilot:

- Ten tasks, three seeds, four model rows, `50000` steps
- Start with packed GPU training and CPU-only evaluation
- Keep array tasks below `3` hours with checkpointing/requeue. If a row times
  out, resume from its checkpoint rather than restarting.

GPU utilization strategy:

- These MLP/LISTA controlled KAEs are too small for one process per GPU.
  Training jobs should pack multiple independent rows on one GPU.
- Use `BATCH_SIZE=512` or `1024` and `PACK_CONCURRENCY=4` initially.
- Record GPU telemetry for every array element and summarize mean utilization,
  active fraction, and max memory after completion.
- Keep evaluation CPU-only unless profiling shows CPU evaluation dominates wall
  time; periodic evaluation is mostly model inference over held-out arrays and
  does not warrant a dedicated GPU for these small models.

## 2026-06-10 Data Preparation Result

Concrete result:

- Smoke job: `9801692`, `PickCube-v1`, `MAX_EPISODES=64`, exit `0:0`, elapsed
  `54` seconds, CPU-only.
- Full data-prep array: `9801694`, CPU-only. Six rows completed on the first
  pass. Four rows failed because their official demos used RL filenames such as
  `trajectory.none.*.h5` rather than `trajectory.h5`.
- Rescue array after trajectory-discovery patch:
  `9801740`, rows `2,3,5,7`, exit `0:0` for all four rows.
- Prepare script:
  [run_maniskill10_prepare_array.sh](/home/mila/l/lia/skae/scripts/run_maniskill10_prepare_array.sh)
- GPU usage: none. All preparation jobs ran on `long-cpu`.

Generated compact datasets:

| Task | Episodes | Max transitions | Obs dim | Action dim | Train/Val/Test |
| --- | ---: | ---: | ---: | ---: | --- |
| `PickCube-v1` | `1000` | `50` | `78` | `8` | `700/150/150` |
| `PushCube-v1` | `1000` | `50` | `78` | `8` | `700/150/150` |
| `PullCube-v1` | `1024` | `50` | `74` | `4` | `717/154/153` |
| `PokeCube-v1` | `761` | `50` | `87` | `4` | `533/114/114` |
| `StackCube-v1` | `1000` | `50` | `78` | `8` | `700/150/150` |
| `RollBall-v1` | `688` | `68` | `74` | `4` | `482/103/103` |
| `PullCubeTool-v1` | `1000` | `100` | `78` | `8` | `700/150/150` |
| `PushT-v1` | `888` | `100` | `82` | `3` | `622/133/133` |
| `PegInsertionSide-v1` | `1000` | `100` | `78` | `8` | `700/150/150` |
| `PlugCharger-v1` | `1000` | `200` | `78` | `8` | `700/150/150` |

Interpretation:

The compact state/action interface is viable for all ten selected default
tasks. The only caveat is `RollBall-v1`: the official downloaded trajectory
coverage reaches `68` transitions even though the manifest endpoint is `80`, so
the H80 endpoint should be treated as unavailable for the current downloaded
packet unless longer demos are found or generated.

## 2026-06-10 Packed GPU Smoke Result

Concrete result:

- Initial GPU submission `9801761` exposed a launcher dependency bug: the GPU
  node lacked `jq`. The pending task was canceled, and the failed task consumed
  about one second with negligible GPU activity.
- Corrected packed GPU smoke `9801812` completed both array tasks with exit
  `0:0`: `PickCube-v1` in `2:14` and `PlugCharger-v1` in `3:41`.
- The smoke wrote all `8/8` expected periodic evaluation summaries under
  [forecast_5k_smoke_9801812](/home/mila/l/lia/skae/runs/maniskill10_default/forecast_5k_smoke_9801812).
- Each array element packed four concurrent training rows on one Quadro RTX
  8000 with `PACK_CONCURRENCY=4`, `BATCH_SIZE=512`, and per-task
  `z_dim=320`.

Best-periodic held-out state MSE:

| Task | Model | Best checkpoint | H10 | H20 | H50 | H100 | H150 | H200 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `PickCube-v1` | dense tanh, no sparsity | `3500` | `1.42e-4` | `1.87e-4` | `1.13e-2` | n/a | n/a | n/a |
| `PickCube-v1` | LISTA ReLU, `alpha=0.01`, `sp=1e-4` | `3500` | `1.62e-4` | `2.02e-4` | `1.07e-2` | n/a | n/a | n/a |
| `PickCube-v1` | sparse MLP ReLU, `sp=0.003` | `500` | `5.03e-4` | `5.97e-4` | `7.57e-3` | n/a | n/a | n/a |
| `PickCube-v1` | sparse MLP ReLU, `sp=0.01` | `500` | `4.06e-4` | `4.93e-4` | `7.45e-3` | n/a | n/a | n/a |
| `PlugCharger-v1` | dense tanh, no sparsity | `4000` | `8.43e-4` | `1.09e-3` | `1.09e-3` | `2.08e-2` | `7.81e-2` | `1.09e-1` |
| `PlugCharger-v1` | LISTA ReLU, `alpha=0.01`, `sp=1e-4` | `4000` | `6.42e-4` | `5.99e-4` | `9.15e-4` | `2.16e-2` | `7.65e-2` | `1.05e-1` |
| `PlugCharger-v1` | sparse MLP ReLU, `sp=0.003` | `4000` | `5.73e-4` | `8.52e-4` | `1.48e-3` | `2.16e-2` | `7.81e-2` | `1.15e-1` |
| `PlugCharger-v1` | sparse MLP ReLU, `sp=0.01` | `4000` | `5.77e-4` | `6.81e-4` | `7.93e-4` | `2.06e-2` | `7.77e-2` | `1.23e-1` |

Coverage note: `PlugCharger-v1` has `150` held-out episodes through H100, but
only `141` eligible episodes at H150 and `38` at H200 in this smoke. H150/H200
are therefore useful diagnostics, not headline endpoint claims from this run.

Ratios versus dense tanh:

| Task | Model | H10 | H20 | H50 | H100 | H150 | H200 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `PickCube-v1` | LISTA ReLU | `1.140` | `1.080` | `0.951` | n/a | n/a | n/a |
| `PickCube-v1` | sparse MLP ReLU, `sp=0.003` | `3.533` | `3.189` | `0.671` | n/a | n/a | n/a |
| `PickCube-v1` | sparse MLP ReLU, `sp=0.01` | `2.850` | `2.634` | `0.661` | n/a | n/a | n/a |
| `PlugCharger-v1` | LISTA ReLU | `0.762` | `0.549` | `0.843` | `1.036` | `0.979` | `0.963` |
| `PlugCharger-v1` | sparse MLP ReLU, `sp=0.003` | `0.679` | `0.782` | `1.361` | `1.037` | `0.999` | `1.062` |
| `PlugCharger-v1` | sparse MLP ReLU, `sp=0.01` | `0.685` | `0.625` | `0.730` | `0.988` | `0.995` | `1.132` |

Support-family diagnostics at threshold `0.2`:

| Task | Model | Families | Mean support size | Outcome NMI | Contact-phase NMI |
| --- | --- | ---: | ---: | ---: | ---: |
| `PickCube-v1` | dense tanh | `26` | `168.5` | `0.008` | `0.015` |
| `PickCube-v1` | LISTA ReLU | `13` | `186.5` | `0.021` | `0.011` |
| `PickCube-v1` | sparse MLP ReLU, `sp=0.003` | `3` | `199.3` | `0.000` | `0.034` |
| `PickCube-v1` | sparse MLP ReLU, `sp=0.01` | `34` | `166.5` | `0.065` | `0.022` |
| `PlugCharger-v1` | dense tanh | `35` | `179.8` | `0.132` | `0.044` |
| `PlugCharger-v1` | LISTA ReLU | `1` | `213.5` | `0.000` | `0.000` |
| `PlugCharger-v1` | sparse MLP ReLU, `sp=0.003` | `153` | `137.9` | `0.242` | `0.075` |
| `PlugCharger-v1` | sparse MLP ReLU, `sp=0.01` | `463` | `75.1` | `0.256` | `0.086` |

Resource and GPU-utilization audit:

- CPU data preparation used no GPUs.
- Corrected smoke `9801812` used two short one-GPU array elements. Combined
  telemetry across `71` samples: mean utilization `58.0%`, max utilization
  `100%`, active fraction with utilization at least `50%` was `57.7%`, and max
  memory was `1476` MiB.
- `PickCube-v1` utilization was good for this small model class: mean `77.4%`,
  max `100%`, active fraction `77.8%`.
- `PlugCharger-v1` hit `100%` during the packed training phase but spent about
  half the short allocation in low-GPU periodic evaluation and support
  bookkeeping: mean `46.0%`, active fraction `45.5%`.

Explanation:

The two-task smoke verifies that the default-task compact datasets, overcomplete
state/action KAE training path, periodic evaluation path, and four-row GPU
packing all run end to end. Forecasting is promising but task dependent. On
`PlugCharger-v1`, sparse rows beat dense tanh at short horizons and remain
competitive through H150/H200. On `PickCube-v1`, LISTA is close to dense tanh
at short horizons and slightly better at H50, while sparse MLP rows lag at
short horizons but win at H50.

Interpretation:

This is enough to justify a larger ManiSkill-10 pilot only after improving the
resource plan. The current GPU training phase is efficient when four rows are
packed, but the integrated periodic evaluation/support stage can hold a GPU
while doing mostly low-utilization work on longer tasks. The next paper-facing
run should either split evaluation/support into CPU jobs after GPU training or
make the GPU training array write checkpoints only and submit dependent CPU
evaluation arrays.

Implementation update after the smoke:

- [run_maniskill10_5k_forecast_smoke_gpu.sh](/home/mila/l/lia/skae/scripts/run_maniskill10_5k_forecast_smoke_gpu.sh)
  now defaults to GPU training only. It skips evaluation unless
  `RUN_EVAL_IN_GPU_JOB=1`, and it refuses CUDA evaluation unless
  `ALLOW_GPU_EVAL=1`.
- [run_maniskill10_eval_cpu_array.sh](/home/mila/l/lia/skae/scripts/run_maniskill10_eval_cpu_array.sh)
  is the CPU-only dependent evaluation/support launcher for saved checkpoints.

Project implications:

The default-task stream is viable as a representation-quality forecasting
benchmark. It should not be used as the main support-regime discovery evidence:
default demos have weak or low-entropy labels, and the support-family metrics
above are exploratory only. The strongest immediate forecasting signal is on
the harder longer-horizon `PlugCharger-v1`, where sparse MLP and LISTA both
show short-horizon gains versus dense tanh.

Next steps:

- Refactor the ManiSkill-10 launcher into GPU training plus dependent CPU
  periodic-evaluation/support jobs before scaling beyond this smoke. The
  scripts now support this split; use that split for scale-up.
- Then queue the ten-task, three-seed pilot with the same four rows and `50000`
  training steps, keeping GPU jobs packed and checkpointed.
- Keep `RollBall-v1` H68 as the endpoint for the current downloaded packet
  unless longer demos are found or generated.
