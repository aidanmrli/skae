# Controlled Basin-Transfer Support Switching

Date: 2026-04-23

## Audit Status

The first seed-`0` fixed-`17` output under
`results/controlled_transfer_switching_fixed17_seed0_20260423` is useful as a
smoke-scale diagnostic, but the summary should be treated as
**procedure-suspect**. The first corrected output under
`results/controlled_transfer_switching_fixed17_seed0_20260423_corrected`
fixed the row-level labeling and post-bridge interpretation risks below, but a
second audit found one remaining summary-accounting issue: for transfer rows,
`post_target`, `post_phase_target`, lag, premature-switch, and chatter metrics
were computed on distinct source/target object rows, while `pre_source` was
still averaged over all transfer rows. The current paper-facing packet is the
second corrected rerun under
`results/controlled_transfer_switching_fixed17_seed0_20260424_reaudit`.

The first audit found two mistakes / over-reading risks:

1. If a native `basin_label` method existed but failed on the bridge trajectory
   tensor, the evaluator skipped the row instead of falling back to
   endpoint-rollout nearest-center labels. The corrected evaluator validates
   native label shape and falls back to endpoint labels when native labeling
   fails.
2. The generated summary emphasized `post_target_dominance`, measured from the
   first target-basin entry onward. In a state-bridge intervention, target
   entry can occur during the synthetic bridge segment before release into
   unforced dynamics. The corrected summary surfaces
   `post_phase_target_dominance` and `post_phase_crossing_lag_steps`, and
   computes switch metrics on rows where source and target support objects are
   distinct.

The existing row-level artifacts already contain `post_phase_target_dominance`
and `source_target_same_object`, but the old summary made it too easy to read
same-object or bridge-dominated rows as clean switching evidence.

The second corrected rerun completed as SLURM jobs `9347926-9347928` with
`1,776` total rows, `1,632` ok rows, `144` skipped rows, and `0` failures.

Second corrected result:

- Dense LISTA exact `topk:8` supports show the strongest support-switching
  evidence: transfer pre-source dominance `0.8194`, post-target dominance
  `0.8230`, post-bridge target dominance `0.9370`, post-bridge lag `6.0455`
  steps, and chatter `0.0375`.
- The zero-sparsity MLP exact `topk:8` supports are weak on the same read:
  pre-source `0.3710`, post-target `0.3114`, post-bridge target `0.3504`,
  post-bridge lag `9.0455`, and no-transfer false-switch rate `0.5178`.
- Blockdiag LISTA exact `topk:8` supports collapse as switch objects:
  pre-source `0.0172`, post-target `0.0448`, and post-bridge target `0.0519`.
- Support-family `topk:8` switching is clean for all three roots. Post-bridge
  target dominance is `1.0000` for dense LISTA, `0.9989` for blockdiag LISTA,
  and `1.0000` for the zero-sparsity MLP.
- Under the stricter summary accounting, family-level transfer pre-source
  dominance is computed only on distinct source/target object rows:
  `0.9555` for dense LISTA `topk:8`, `0.9060` for blockdiag LISTA `topk:8`,
  and `0.9124` for the zero-sparsity MLP `topk:8`.

Interpretation:

The paper-positive switching claim is dense LISTA exact `topk:8`, not generic
support-family switching. Family-level switching is useful evidence that the
learned representations track basin changes after a deliberate state-space
bridge, but it is not sufficient for a LISTA-specific sparsity mechanism
because the zero-sparsity MLP family read is also strong. This remains a
state-space intervention diagnostic, not an admissible optimal-control transfer
experiment.

## Purpose

This first-pass branch tests the missing reviewer-facing mechanism:

`basin -> sparse support -> local linear law`

Existing fixed-17 packets show support persistence, support families, centered local-law structure, and self-routed forecasting. They do not yet show that when a trajectory is deliberately transferred from one basin to another, the model support stays stable in the source basin, switches near the measured basin crossing, and stabilizes in the target basin without chatter.

This branch is evaluation-only. It does not train new models and does not use basin labels for routing. Basin labels are used only to measure source/target membership, source-exit time, target-entry time, transfer success, and benchmark metrics.

## Hypothesis

For a fixed trained checkpoint and a fixed support definition, controlled transfers should show:

1. Source-basin stability: the exact support or support family is stable before the measured basin crossing.
2. Timed switching: the first target-support or target-family dominance occurs near, not far before, the measured target-entry time.
3. Target-basin stabilization: after the measured crossing, the target support or target family dominates with limited chatter.
4. Control specificity: matched no-transfer trajectories from the same source depth do not show comparable false switching.

The most interpretable paper-positive result is a support object that is both
stable within a basin and distinct across transferred basins. Exact `topk:8`
supports are therefore the strictest read; support families are more robust
but can become too coarse to isolate a LISTA-specific sparsity mechanism.
`relative:0.1` is optional and mainly a robustness/diagnostic support
definition.

## Falsifiable Outcomes

The result should be treated as negative or inconclusive if:

1. Supports switch long before the measured target-entry time on transfer trajectories.
2. Supports do not switch after successful source-to-target transfers.
3. Post-crossing support/family identity chatters instead of stabilizing.
4. Matched no-transfer controls have similar false-switch rates.
5. Most attempted transfers fail to reach the intended target basin.
6. Results only appear on transfers whose source-exit or target-entry time cannot be measured reliably.
7. The source and target reference objects are identical for many pairs; this
   is evidence against switch-identifiability for that support object, not a
   positive switching result.

Failed physical transfers are separated from support-switch failures. A trajectory that does not actually move from source to target is not evidence against support switching; it is a transfer-construction failure.

## Systems Included And Excluded

Included by default:

1. The fixed-17 transition-rich systems from `skae/benchmarks/transition_rich_basin_partition_manifest.py`.
2. Systems where the evaluator can extract attractor or basin-center proxies from the environment.
3. Systems where benchmark labels can be measured either by native `basin_label(state)` or by long rollout to extracted centers.

Excluded or skipped with reasons:

1. Systems with no usable centers or endpoint proxies.
2. Systems whose extracted center count is smaller than the manifest basin count.
3. Systems where bridge states cannot be labeled after long rollout.
4. Transfer pairs where the source phase is not actually source-labeled.
5. Transfer pairs where the final state does not label as the target.

This is deliberately conservative. The evaluator should not fabricate controlled-transfer evidence on environments whose API does not support evaluating the intervention.

## Transfer Construction

The first-pass construction is a deterministic state-space bridge intervention:

1. Select source and target basin centers from `env.points` if available, otherwise from Claude-catalog center-like attributes such as `_wells`, `wells`, `well_centers`, `centers`, `room_centers`, or `dipoles`.
2. Start at a matched source depth along the line from source center to target center.
3. Run a short unforced source phase using `env.step`.
4. Apply a linear state bridge from the source-side state toward a matched target-side entry point.
5. Release the trajectory and run an unforced target phase using `env.step`.
6. Build a matched no-transfer control from the same source initial condition and same total duration, using only `env.step`.

This is recorded as `state_bridge_linear_intervention`. It is not a claim that the unforced system naturally crosses, and it is not an admissible optimal-control trajectory unless an environment explicitly exposes compatible control inputs. It is a deliberate state-space intervention designed to ask whether the encoder support updates when the benchmark basin membership changes.

## Confounder Controls

1. Same checkpoint and model are used for all support definitions and controls.
2. Same support definitions are used across systems: default `absolute:0.001,topk:8`; optional `relative:0.1`.
3. Same source-depth fraction is used for all source-target pairs.
4. Matched no-transfer controls use the same source initial state and same total trajectory length.
5. Transfer success is recorded separately from support-switch quality.
6. Basin labels are never used to choose support routes or target supports. They are used only for measurement.
7. Target support references are computed from a target-entry unforced reference trajectory, not from oracle routing.
8. Support-family labels are assigned jointly across the transfer trajectory, target-reference trajectory, and matched no-transfer control for each comparison, so the same family ID refers to the same support prototype within that row.
9. Multi-basin transfers distinguish `source_exit_index` from `target_entry_index`. Target-support lag is measured relative to target entry, not merely first departure from the source basin, because a state bridge may pass through an intermediate basin.
10. Rows explicitly flag `source_target_same_object`. A row where source and
   target references share the same support/family cannot support a switching
   claim, even if post-target dominance is high.

## Metrics

Per successful transfer and per support object (`support`, `family`):

1. `pre_source_dominance`: fraction of pre-source-exit states equal to the source object.
2. `post_target_dominance`: fraction of post-target-entry states equal to the target reference object.
3. `crossing_lag_steps`: first target-object time minus measured target-entry time.
4. `source_exit_to_target_entry_steps`: delay between leaving the source basin and first entering the target basin.
5. `premature_switch_rate`: fraction of pre-target-entry states already equal to the target object.
6. `source_target_same_object`: whether source and target reference objects are identical.
7. `switch_interpretation_status`: whether the row can be interpreted as a source-target support switch.
8. `chatter_switch_rate`: support-object switch fraction over the whole transfer.
9. `post_chatter_switch_rate`: support-object switch fraction after target entry.
10. `post_phase_target_dominance`: target-object dominance after the bridge has
    ended and the trajectory is in the unforced target phase.
11. `post_phase_crossing_lag_steps`: first target-object time measured from
    the unforced target-phase start.
12. `support_entropy` and `support_entropy_normalized`: entropy of support-object labels along the trajectory.
13. `false_switch_rate` on no-transfer controls: fraction of post-source states not equal to the source object.
14. Coverage fields: `status`, `skip_reason`, `transfer_success`, `source_exit_index`, `target_entry_index`, `crossing_index`, `source_pre_fraction`, `final_target_fraction`.

## Expected Interpretation

A clean positive result would show high pre-source dominance before source exit, high post-target dominance after target entry, small nonnegative target-entry lag, low premature switching before target entry, and low no-transfer false switching. That would directly support the claim that support families behave like state-dependent regime objects under deliberate basin transfer.

A mixed result can still be useful. For example, exact supports may be brittle while support families switch cleanly. That would favor a paper narrative around support-family alignment rather than one exact support per basin.

## Commands

Smoke-size SLURM run on one root/system/seed:

```bash
ROWS_CSVS=results/transition_rich_basin_partition_final_seed10_20260409/collect_pass0/forecasting_rows.csv \
OUT_DIR=results/controlled_transfer_switching_smoke_20260423 \
ROOT_LABELS_CSV=lista_dense_softblock_signsplit_p64_hardinit_basin_partition \
SYSTEMS_CSV=claude:cal_square_4 \
SEEDS_CSV=0 \
SMOKE=1 \
sbatch scripts/run_transition_rich_controlled_transfer_switching.sh
```

Full first-pass run:

```bash
ROWS_CSVS=results/transition_rich_basin_partition_final_seed10_20260409/collect_pass0/forecasting_rows.csv \
OUT_DIR=results/controlled_transfer_switching_20260423 \
ROOT_LABELS_CSV=lista_dense_softblock_signsplit_p64_hardinit_basin_partition,lista_blockdiag_signsplit_hardinit_basin_partition \
SUPPORT_DEFINITIONS=absolute:0.001,topk:8,relative:0.1 \
sbatch scripts/run_transition_rich_controlled_transfer_switching.sh
```

Queued shard run:

```bash
ROWS_CSVS=results/transition_rich_basin_partition_final_seed10_20260409/collect_pass0/forecasting_rows.csv \
OUT_DIR=results/controlled_transfer_switching_20260423 \
ROOT_LABELS_CSV=lista_dense_softblock_signsplit_p64_hardinit_basin_partition,lista_blockdiag_signsplit_hardinit_basin_partition \
SEED_SPLITS_SEMICOLON='0;1;2' \
scripts/queue_transition_rich_controlled_transfer_switching.sh
```

## Outputs

The evaluator writes:

1. `controlled_transfer_switching_rows.csv`
2. `controlled_transfer_switching_summary.md`
3. `manifest.json`
4. `failures.json`
5. `progress.json`

## First-Pass Limitations

1. The bridge is an explicit state intervention, not a naturally occurring transfer and not an optimal-control/admissible-control trajectory in the control-theoretic sense.
2. Claude-catalog basin centers are extracted heuristically from known center-like attributes.
3. Systems without reliable center extraction or endpoint labels are skipped.
4. The code has not been executed in this handoff pass; it must be smoke-tested on a compute node before paper-facing results are reported.
