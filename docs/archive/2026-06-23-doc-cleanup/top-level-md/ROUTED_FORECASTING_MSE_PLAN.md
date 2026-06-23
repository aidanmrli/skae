# Routed Forecasting MSE Plan

Date: May 6, 2026

## Current Objective

Train support-family-selected local Koopman maps \(K_c\) long enough to test
whether label-free sparse support families can become competitive forecasters
against the original global \(K\) in the existing LISTA table models.

The fixed setting is selected from the completed one-seed controlled
proof-of-concept:

- Model family: dense LISTA.
- Controlled multibasin root:
  `lista_dense_signsplit_p256_hardinit_basin_partition`.
- Dysts root: `lista`.
- Support definition: `F_top8`.
- Family merge threshold: Jaccard `0.40`.
- Re-encoding cadence: every `5` steps.
- Routing mode: `reroute_each_step`.
- Trainable component: route-local centered maps \(K_c\) only.
- Frozen components: encoder, decoder, and original global \(K\).
- Initialization: every \(K_c\) starts from the original global \(K\).
- Training budget: first report `50000` route-balanced optimizer steps, then
  continue the same \(K_c\) checkpoints to `100000` total steps and report
  the 100k results.
- Worker budget: `03:00:00` on `long` with one GPU, one CPU, and `8G` memory.

This is not a new tuning sweep. It is a fixed-setting multi-seed confirmation
of the best one-seed local-routing recipe.

## \(F_{\rm abs}\) Router-Comparison Replay

To support the direct \(F_{\rm abs}\) versus \(F_{\rm top8}\) audit, a matching
`absolute:0.001` stage-2 replay was run against the existing `topk:8`
fixed-setting results:

- Support definition: `absolute:0.001`.
- Family merge threshold: Jaccard `0.40`.
- Re-encoding cadence: every `5` steps.
- Routing mode: `reroute_each_step`.
- Training budget: `50000` route-balanced optimizer steps.
- Worker budget: `03:00:00` on `long` with one GPU, one CPU, and `8G` memory.
- Checkpointing: each shard writes `train_checkpoint.pt` during progress
  logging, plus `metrics_history.jsonl`, `local_maps.pt`, `route_codebook.json`,
  `stage2_config.json`, and routed-forecast rows.

Submitted 50k batches:

| Benchmark/Seeds | Worker jobs | Merge | Combined job | Output directory |
|---|---|---:|---:|---|
| Controlled seeds `0`--`4` | `9478281`--`9478356` | `9478357` | `9478541` | [fabs_lista_multibasin_abs001_j040_50k_seed0_4](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/fabs_lista_multibasin_abs001_j040_50k_seed0_4) |
| Controlled seeds `5`--`9` | `9478359`--`9478433` | `9478434` | `9478541` | [fabs_lista_multibasin_abs001_j040_50k_seed5_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/fabs_lista_multibasin_abs001_j040_50k_seed5_9) |
| Dysts seeds `0`--`4` | `9478436`--`9478485` | `9478486` | `9478542` | [fabs_lista_dysts_abs001_j040_50k_labelnone_seed0_4](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/fabs_lista_dysts_abs001_j040_50k_labelnone_seed0_4) |
| Dysts seeds `5`--`9` | `9478488`--`9478537` | `9478538` | `9478542` | [fabs_lista_dysts_abs001_j040_50k_labelnone_seed5_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/fabs_lista_dysts_abs001_j040_50k_labelnone_seed5_9) |

Result:

- All `250` shards wrote `train_checkpoint.pt`, so checkpointing worked.
- Controlled multibasin is not aggregate-usable: `149/150` workers ended
  `OUT_OF_MEMORY`, leaving only `claude:snic_multi` seed `6` completed. The
  partial controlled combine failed because the seed-`0`--`4` batch had no
  nonempty merged row file.
- Dysts completed `96/100` workers. The four OOMs were Shimizu-Morioka seeds
  `0`, `1`, and `6`, plus Chua seed `9`.
- Partial Dysts aggregate:
  [routed_forecasting_iqm_summary.csv](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/combined_fabs_lista_dysts_abs001_j040_50k_labelnone_seed0_9_partial/aggregation/routed_forecasting_iqm_summary.csv).
  Route coverage is `0.2345`, fallback is `0.7655`, \(F_{\rm abs}\) wins
  `0/10` systems versus best-periodic at every horizon, and routed/best-periodic
  ratios are `6.39e3` (`H100`), `5.37e27` (`H500`), `5.22e25` (`H1000`),
  `1.56e23` (`H1500`), `7.53e15` (`H2000`), `1.69e23` (`H3000`), `9.43e26`
  (`H4000`), and `6.74e4` (`H5000`).

## Stage-2 Calibrated Global \(K\) Ablation

Add a fairness ablation that uses the same second-stage data and rollout-MSE
objective, but learns one dense global map instead of one map per support
family:

- Mode: `stage2_map_mode=global_dense_calibrated`.
- Initialization: the trainable map \(K_{\mathrm{cal}}\) starts from the
  original checkpoint global \(K\).
- Frozen components: encoder and decoder remain frozen.
- Trainable component: only \(K_{\mathrm{cal}}\).
- Training data: same generated short training-window pool as the \(K_c\) run.
- Sampling: same route-balanced minibatch sampler, using the same `F_top8`,
  `J=0.40` codebook only to balance examples across support families.
- Prediction rule: \(\hat z_{t+1}=K_{\mathrm{cal}}\hat z_t\); the selected
  support family is ignored by the transition rule.
- Evaluation: same periodic decode/re-encode cadence, horizons, and
  aggregation as the \(K_c\) run.

This ablation separates the effect of **stage-2 rollout calibration** from the
effect of **support-family-local parameterization**. The three relevant rows
are:

1. Frozen original global \(K\): already present in the paper-table
   forecasting runs.
2. Stage-2 calibrated global \(K_{\mathrm{cal}}\): queued here.
3. Stage-2 support-family local \(K_c\): the current primary local-routing
   run.

## Activation Alignment

The local-map training must match the sparse code used by the corresponding
LISTA table checkpoints:

- Controlled multibasin uses the p256 dense LISTA checkpoint with
  `FINAL_OP=sign_split`.
- Dysts uses the p256 dense LISTA checkpoint with `FINAL_OP=relu`.

This keeps each local-routed result aligned with the original global-\(K\)
LISTA baseline for the same benchmark. The controlled and Dysts packets still
have different final sparse-code nonlinearities, so cross-benchmark claims
should be phrased as benchmark-matched local-vs-global comparisons rather than
as one shared activation experiment.

## Route Construction

- Support rule: `topk:8`.
- Family construction: greedy Jaccard support-family merging at threshold
  `0.40`.
- Minimum fitting transitions: `50`.
- Families with fewer than `50` fitting transitions fall back to the frozen
  global \(K\).
- No basin labels, attractor labels, or known basin counts are used to build
  routes, fit route families, train \(K_c\), or select \(K_c\).
- Basin labels remain evaluation-only diagnostics on controlled benchmarks.

For each fitted family \(c\), the representative center \(\bar z_c\) is the
mean encoded latent over fitting transitions assigned to that family. The local
rule is
\[
\hat z_{t+1} = \bar z_c + K_c(\hat z_t - \bar z_c).
\]

## Training Loop

Detailed protocol and pseudocode:
[SUPPORT_FAMILY_LOCAL_KC_TRAINING_PROTOCOL.md](/home/mila/l/lia/skae/docs/SUPPORT_FAMILY_LOCAL_KC_TRAINING_PROTOCOL.md).

For a training window \((x_0,\ldots,x_H)\):

1. Encode \(z_0=\mathrm{Enc}(x_0)\).
2. Select the current support family using `F_top8` and `J=0.40`.
3. Apply the selected centered map \(K_c\), or global \(K\) for fallback
   families.
4. Decode the predicted latent.
5. Every `5` steps, decode/re-encode the prediction to return to the encoder
   manifold.
6. Because the selected mode is `reroute_each_step`, recompute the support
   family before every local-map application.
7. Minimize decoded rollout MSE over the original short training horizon:
   `8` for controlled multibasin and `10` for Dysts.

The minibatch sampler is route-balanced: each optimizer step samples routes
approximately uniformly and then samples windows within the selected routes.
This prevents frequent families from monopolizing the local-map updates.

## Queue Plan

Queue four fixed-setting batches:

1. Controlled multibasin seeds `0,1,2,3,4` on the retained `15` systems.
2. Dysts seeds `0,1,2,3,4` on the retained `10` systems.
3. Controlled multibasin seeds `5,6,7,8,9` on the retained `15` systems.
4. Dysts seeds `5,6,7,8,9` on the retained `10` systems.

Each worker handles exactly one `(benchmark, system, seed, period, routing
mode)` shard. This avoids putting five seeds into one three-hour job and makes
resubmission/checkpoint recovery local to a single seed-system pair.

Submission status:

| Batch | Worker jobs | Merge | Aggregation | Output directory |
|---|---|---:|---:|---|
| Controlled seeds `0`--`4` | `9474335`--`9474409` | `9474410` | `9474411` | [best_lista_multibasin_j040_50k_seed0_4](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/best_lista_multibasin_j040_50k_seed0_4) |
| Dysts seeds `0`--`4` | `9474413`--`9474462` | `9474463` | `9474464` | [best_lista_dysts_j040_50k_seed0_4](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/best_lista_dysts_j040_50k_seed0_4) |
| Controlled seeds `5`--`9` | `9474465`--`9474539` | `9474540` | `9474541` | [best_lista_multibasin_j040_50k_seed5_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/best_lista_multibasin_j040_50k_seed5_9) |
| Dysts seeds `5`--`9` | `9474542`--`9474591` | `9474592` | `9474593` | [best_lista_dysts_j040_50k_seed5_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/best_lista_dysts_j040_50k_seed5_9) |

Compile checks `9474328` and `9474331` completed with exit `0:0` before the
full launch; follow-up checkpoint-resume compatibility compile `9474598` also
completed with exit `0:0`. The latest live queue check showed `15` workers
running and `243` queued/pending dependency jobs, with `0` nonempty worker
stderr files under the 50k output tree. Worker `9474335` started on a V100 and
reached at least step `9500/50000` for controlled `claude:arrested_spiral`,
seed `0`; `15` shard-level `train_checkpoint.pt` files had already been
written.

Status correction: the first Dysts submissions used the default
`LABEL_MODE=auto`, which is correct for controlled systems but invalid for
Dysts and produced `Unknown transition-rich basin-partition system` failures.
The pending Dysts seed `5`--`9` jobs from that first submission were cancelled,
and Dysts was resubmitted with `LABEL_MODE=none` into clean label-free output
directories:

| Corrected Dysts batch | Worker jobs | Merge | Aggregation | Output directory |
|---|---|---:|---:|---|
| Dysts seeds `0`--`4` | `9475039`--`9475088` | `9475089` | `9475090` | [best_lista_dysts_j040_50k_labelnone_seed0_4](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/best_lista_dysts_j040_50k_labelnone_seed0_4) |
| Dysts seeds `5`--`9` | `9475091`--`9475140` | `9475141` | `9475142` | [best_lista_dysts_j040_50k_labelnone_seed5_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/best_lista_dysts_j040_50k_labelnone_seed5_9) |

Combined 10-seed 50k evaluations:

| Benchmark | Dependency | Combined aggregation job | Output directory |
|---|---|---:|---|
| Controlled multibasin seeds `0`--`9` | after seed `5`--`9` merge `9474540` (`9474410` already complete) | `9475184` | [combined_best_lista_multibasin_j040_50k_seed0_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/combined_best_lista_multibasin_j040_50k_seed0_9) |
| Dysts seeds `0`--`9` | after corrected Dysts merges `9475089` and `9475141` | `9475178` | [combined_best_lista_dysts_j040_50k_labelnone_seed0_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/combined_best_lista_dysts_j040_50k_labelnone_seed0_9) |

After these combined 50k aggregations complete, report the 10-seed controlled
and Dysts results. Dependency-held 100k launchers are already queued behind
the same combined 50k aggregation jobs so the continuation workers are not
submitted until the 50k evaluation artifact exists.

## 100k Continuation Plan

The 100k run should continue the same \(K_c\) maps from the 50k
`train_checkpoint.pt` artifacts and write into new result directories. It
must not overwrite the 50k directories.

Continuation settings:

- `TRAIN_STEPS=100000` means total optimizer steps, not 100k additional steps.
- `RESUME_FROM_OUTPUT_DIRS` points to the matching 50k batch output directory.
- If a matching `train_checkpoint.pt` is found, the trainer resumes from its
  recorded `next_step`; otherwise the shard starts from global-\(K\)
  initialization and should be treated as misconfigured.
- Controlled uses `LABEL_MODE=auto`; Dysts uses `LABEL_MODE=none`.
- Evaluate the 100k continuation with the same forecast horizons as 50k.

Planned 100k output directories:

- `results/routed_stage2_local_maps_20260506/best_lista_multibasin_j040_100k_from50k_seed0_4`
- `results/routed_stage2_local_maps_20260506/best_lista_multibasin_j040_100k_from50k_seed5_9`
- `results/routed_stage2_local_maps_20260506/best_lista_dysts_j040_100k_from50k_labelnone_seed0_4`
- `results/routed_stage2_local_maps_20260506/best_lista_dysts_j040_100k_from50k_labelnone_seed5_9`

Then combine the two controlled 100k batches and the two Dysts 100k batches
with [scripts/combine_and_analyze_stage2_batches.sh](/home/mila/l/lia/skae/scripts/combine_and_analyze_stage2_batches.sh),
using the same seed-IQM-within-system and arithmetic-mean-over-systems
aggregation.

100k continuation launch status:

| Benchmark | Launcher dependency | Launcher job | Expected combined output |
|---|---|---:|---|
| Controlled multibasin seeds `0`--`9` | after 50k combined aggregation `9475184` | `9475264` | [combined_best_lista_multibasin_j040_100k_from50k_seed0_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/combined_best_lista_multibasin_j040_100k_from50k_seed0_9) |
| Dysts seeds `0`--`9` | after 50k combined aggregation `9475178` | `9475265` | [combined_best_lista_dysts_j040_100k_from50k_labelnone_seed0_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/combined_best_lista_dysts_j040_100k_from50k_labelnone_seed0_9) |

The launcher script submits the two seed batches for a benchmark and then
submits the benchmark-level 10-seed combine/aggregation job after the two 100k
batch merges. This preserves the requested order: 50k aggregation first, then
100k continuation from the matching `train_checkpoint.pt` files, then 100k
aggregation.

## Calibrated Global \(K\) Queue Status

Compile check `9476978` completed with exit `0:0`. The calibrated-global
ablation is queued with the same fixed setting as the primary \(K_c\) run:
`F_top8`, `J=0.40`, `period=5`, `reroute_each_step`, one GPU, one CPU, `8G`,
and `03:00:00` on `long`.

50k calibrated-global batches:

| Benchmark/Seeds | Worker jobs | Merge | Aggregation | Output directory |
|---|---|---:|---:|---|
| Controlled seeds `0`--`4` | `75` jobs, `9476981`--`9477236` interleaved | `9477237` | `9477238` | [globalK_calibrated_multibasin_j040_50k_seed0_4](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/globalK_calibrated_multibasin_j040_50k_seed0_4) |
| Controlled seeds `5`--`9` | `75` jobs, `9476982`--`9477229` interleaved | `9477231` | `9477233` | [globalK_calibrated_multibasin_j040_50k_seed5_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/globalK_calibrated_multibasin_j040_50k_seed5_9) |
| Dysts seeds `0`--`4` | `50` jobs, `9476984`--`9477175` interleaved | `9477178` | `9477181` | [globalK_calibrated_dysts_j040_50k_labelnone_seed0_4](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/globalK_calibrated_dysts_j040_50k_labelnone_seed0_4) |
| Dysts seeds `5`--`9` | `50` jobs, `9476983`--`9477189` interleaved | `9477191` | `9477194` | [globalK_calibrated_dysts_j040_50k_labelnone_seed5_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/globalK_calibrated_dysts_j040_50k_labelnone_seed5_9) |

Combined 50k calibrated-global aggregations:

| Benchmark | Dependency | Combined job | Output directory |
|---|---|---:|---|
| Controlled multibasin seeds `0`--`9` | after merges `9477237` and `9477231` | `9477239` | [combined_globalK_calibrated_multibasin_j040_50k_seed0_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/combined_globalK_calibrated_multibasin_j040_50k_seed0_9) |
| Dysts seeds `0`--`9` | after merges `9477178` and `9477191` | `9477240` | [combined_globalK_calibrated_dysts_j040_50k_labelnone_seed0_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/combined_globalK_calibrated_dysts_j040_50k_labelnone_seed0_9) |

100k calibrated-global continuation launchers:

| Benchmark | Launcher dependency | Launcher job | Expected combined output |
|---|---|---:|---|
| Controlled multibasin seeds `0`--`9` | after 50k combined job `9477239` | `9477244` | [combined_globalK_calibrated_multibasin_j040_100k_from50k_seed0_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/combined_globalK_calibrated_multibasin_j040_100k_from50k_seed0_9) |
| Dysts seeds `0`--`9` | after 50k combined job `9477240` | `9477243` | [combined_globalK_calibrated_dysts_j040_100k_from50k_labelnone_seed0_9](/home/mila/l/lia/skae/results/routed_stage2_local_maps_20260506/combined_globalK_calibrated_dysts_j040_100k_from50k_labelnone_seed0_9) |

Smoke status: the first tiny controlled smoke `9477245` used too few
trajectories for controlled `LABEL_MODE=auto` basin-label diagnostics and
failed before exercising the intended path. Corrected smoke `9477366` used
`LABEL_MODE=none`, completed with exit `0:0`, wrote one
`global_dense_calibrated` row, and recorded `0` failures.

Controlled multibasin systems:

- `claude:arrested_spiral`
- `claude:cal_asymmetric_3`
- `claude:cal_hexagon_6`
- `claude:cal_high_cross_3`
- `claude:cal_octagon_8`
- `claude:cal_pentagon_5`
- `claude:cal_square_4`
- `claude:duffing_triple_well`
- `claude:snic_multi`
- `claude:transition_routes_4`
- `claude:var_depth_gradient_4`
- `claude:var_diamond_4`
- `claude:var_l_shape_5`
- `gated_local_linear`
- `gated_transfer_linear`

Dysts systems:

- `dysts:Chua`
- `dysts:Dadras`
- `dysts:DequanLi`
- `dysts:Hadley`
- `dysts:LuChenCheng`
- `dysts:QiChen`
- `dysts:Sakarya`
- `dysts:SanUmSrisuchinwong`
- `dysts:ShimizuMorioka`
- `dysts:WangSun`

## Outputs

Each shard writes:

- `local_maps.pt`
- `global_map.pt` for `stage2_map_mode=global_dense_calibrated`
- `train_checkpoint.pt`
- `route_codebook.json`
- `route_balancing_codebook.json` for `stage2_map_mode=global_dense_calibrated`
- `stage2_config.json`
- `metrics_history.jsonl`
- `self_routed_forecasting_rows.csv`
- `failures.json`

Each batch writes a merged `self_routed_forecasting_rows.csv`, an aggregation
readout, and a queue manifest. The primary aggregation remains seed IQM within
each system followed by arithmetic mean across systems.

Forecast horizons:

- Controlled multibasin: `H100,H500,H1000`.
- Dysts: `H100,H500,H1000,H1500,H2000,H3000,H4000,H5000`.

## Implementation Notes

- [docs/SUPPORT_FAMILY_LOCAL_KC_TRAINING_PROTOCOL.md](/home/mila/l/lia/skae/docs/SUPPORT_FAMILY_LOCAL_KC_TRAINING_PROTOCOL.md)
  gives the explicit stage-2 \(K_c\) training protocol, including what is
  frozen, what is optimized, the decoded-MSE objective, and pseudocode.
- [tools/train_support_family_local_maps.py](/home/mila/l/lia/skae/tools/train_support_family_local_maps.py)
  trains either the route-local \(K_c\) maps or the calibrated global
  \(K_{\mathrm{cal}}\) ablation via `--stage2_map_mode`, and checkpoints
  train-loop state during progress writes.
- [scripts/run_support_family_local_maps_stage2.sh](/home/mila/l/lia/skae/scripts/run_support_family_local_maps_stage2.sh)
  runs one SLURM worker shard.
- [scripts/queue_support_family_local_maps_stage2_poc.sh](/home/mila/l/lia/skae/scripts/queue_support_family_local_maps_stage2_poc.sh)
  now submits one worker per seed-system pair and records seed-level shard
  slugs in the queue manifest.
- [scripts/launch_support_family_local_maps_100k_from50k.sh](/home/mila/l/lia/skae/scripts/launch_support_family_local_maps_100k_from50k.sh)
  is the dependency-held launcher for 100k continuation. It is submitted after
  the 50k combined aggregation, queues the two seed batches, and submits the
  final 10-seed combine job.
- [tools/analyze_routed_forecasting_mse.py](/home/mila/l/lia/skae/tools/analyze_routed_forecasting_mse.py)
  aggregates routed local-map MSE against the existing global-\(K\)
  best-periodic baselines.

## Prior One-Seed Selection Result

The completed one-seed controlled proof selected dense LISTA with period `5`
and `reroute_each_step` by geometric mean of aggregate `H100/H500/H1000`
routed MSE:

| Candidate | Period | Routing | H100 | H500 | H1000 | Geomean |
|---|---:|---|---:|---:|---:|---:|
| LISTA dense signsplit | 5 | reroute_each_step | 0.1164 | 0.1918 | 0.2024 | 0.1653 |
| LISTA-BD signsplit | 2 | reroute_each_step | 0.1151 | 0.2036 | 0.2150 | 0.1714 |
| LISTA dense signsplit | 2 | reroute_each_step | 0.1176 | 0.2143 | 0.2248 | 0.1783 |

This selected setting was still behind the original global-\(K\) LISTA
baseline, especially at `H100`, but the gap narrowed at longer horizons. The
50k multi-seed run tests whether a longer local-map training stage closes that
gap.

## Documentation Update Rule

When new results land, update:

1. [docs/EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md)
2. [docs/PAPER_TRACK_STATUS.md](/home/mila/l/lia/skae/docs/PAPER_TRACK_STATUS.md)
3. [docs/PAPER_EXPERIMENT_EVIDENCE_MAP.md](/home/mila/l/lia/skae/docs/PAPER_EXPERIMENT_EVIDENCE_MAP.md)

Report results in this order: concrete result, experiment context,
interpretation, paper implication, and next steps.
