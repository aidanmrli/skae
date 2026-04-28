# Per-basin deep-slice interpretability re-evaluation

Date queued: 2026-04-27.

## Motivation

Table 1 of [`docs/neurips_sparse_koopman_multibasin.tex`](neurips_sparse_koopman_multibasin.tex) (`tab:fixed17_alignment`) reports a *wrong-support-freeze* diagnostic with a `K/13` rather than `K/17` denominator. The reason is structural: under the global top-quartile depth criterion (`_margin_subsets`, `tools/reduce_transition_rich_interpretability_metrics.py:510`), four of the seventeen multibasin systems have a deep slice that contains states from a single basin only:

- `claude_arrested_spiral` (5 basins by design)
- `claude_cal_asymmetric_3` (3 basins)
- `claude_duffing_triple_well` (3 basins)
- `claude_var_l_shape_5` (5 basins)

On these systems one basin's attractor has a much larger margin to the second-nearest centroid than the others, so the global top-quartile filter selects only that basin's states. The wrong-support-freeze diagnostic builds a `{basin → canonical support mask}` dictionary from deep-slice states and requires at least two distinct basins to draw a "wrong" mask; on a single-basin deep slice the test is undefined by construction. This holds across all 7 trained model rows and all 9 support-threshold definitions, confirming it is a property of the system's depth-margin geometry rather than of any encoder.

## Plan

Re-run the interpretability reducer with a **per-basin top-quartile** depth criterion. This admits each basin's relative-deep states (top 25% within its own margin distribution), so every basin contributes to the deep slice and the wrong-support-freeze diagnostic is well-defined for all 17 systems.

This is purely an evaluation re-run: training is not affected. We use the saved checkpoints from the locked final packets and re-evaluate with `--depth_slice_mode per_basin`. Outputs are written to a *new* sibling directory under each packet, leaving the existing global-slice results untouched.

## What changes in the code

- `tools/reduce_transition_rich_interpretability_metrics.py`
  - `_margin_subsets()` gains a `depth_slice_mode` keyword (`"global"` (default, unchanged) or `"per_basin"`) and an optional `basin_labels` argument.
  - A new CLI flag `--depth_slice_mode {global,per_basin}` is plumbed through to the per-spec call site.
- `scripts/reduce_transition_rich_interpretability_metrics.sh`
  - New `DEPTH_SLICE_MODE` env var (default `global`) forwarded as `--depth_slice_mode`.
- `scripts/queue_transition_rich_interpretability_shards.sh`
  - New `DEPTH_SLICE_MODE` env var forwarded into each shard sbatch.
- `scripts/queue_per_basin_deep_eval.sh` *(new)*
  - One-shot launcher that submits the per-basin re-eval for the five boundary-emphasized rows, restricted to the matching root labels.

The default behavior of every existing pipeline is preserved: omitting `DEPTH_SLICE_MODE` keeps the global-quartile slice that drives the current Table 1.

## Sampling regime

All 5 re-evaluated rows are trained under the **same boundary-emphasized hard-init sampling regime** that drives the current Table 1. The eval re-run does not change the training distribution. The two source packets contain the relevant checkpoints:

| Row in Table 1 | Trained root | Source packet |
|---|---|---|
| SLK-BD | `lista_blockdiag_signsplit_hardinit_basin_partition` | `transition_rich_basin_partition_final_seed10_20260409` |
| SLK-SB | `lista_dense_softblock_signsplit_p64_hardinit_basin_partition` | `transition_rich_basin_partition_final_seed10_20260409` |
| Sparse MLP, BD | `mlp_sparse_blockdiag_hardinit_basin_partition_control` | `transition_rich_hardinit_mlp_controls_seed10_20260416` |
| Sparse MLP | `mlp_sparse_hardinit_basin_partition_control` | `transition_rich_hardinit_mlp_controls_seed10_20260416` |
| Dense MLP, no shrink | `mlp_zero_sparse_hardinit_basin_partition_control` | `transition_rich_hardinit_mlp_controls_seed10_20260416` |

Coverage: 5 roots × 17 systems × ~10 seeds.

## Output paths (do not overwrite existing results)

- `results/transition_rich_basin_partition_final_seed10_20260409/interpretability_per_basin_deep_pass1/` — SLK-BD, SLK-SB.
- `results/transition_rich_hardinit_mlp_controls_seed10_20260416/interpretability_per_basin_deep_pass1/` — boundary MLP controls.

The existing `interpretability_final_pass1/` directories under both packets remain the source-of-truth for the global-slice numbers in the current Table 1.

## How to launch (and re-launch)

```bash
bash scripts/queue_per_basin_deep_eval.sh
```

The script:
1. Submits one sbatch shard per root via `scripts/queue_transition_rich_interpretability_shards.sh` (default partition: `long-cpu`, 8 hours, 16 GB).
2. Submits a dependent merge job per packet that consolidates shard outputs into a single `interpretability_rows.csv`.
3. Writes a `queue_manifest.json` per packet with shard / merge job IDs.

Per CLAUDE.md cluster policy, this routes through `sbatch` rather than running on the login node.

## Verification before launch

A 1-system / 1-seed dry run on a `salloc`-allocated compute node is a low-cost sanity check that the `--depth_slice_mode per_basin` path produces non-degenerate output for the four problem systems. Suggested:

```bash
salloc --mem=16G -c 4 --partition=long
ROWS_CSV="${PACKET_FINAL}/collect_pass1/forecasting_rows.csv" \
OUT_DIR="/tmp/per_basin_smoke" \
ROOT_LABELS_CSV="lista_dense_softblock_signsplit_p64_hardinit_basin_partition" \
SYSTEMS_CSV="claude:duffing_triple_well" \
SEEDS_CSV="0" \
DEPTH_SLICE_MODE="per_basin" \
  bash scripts/reduce_transition_rich_interpretability_metrics.sh
```

Expect: `support_freeze_template_count >= 2` on all four previously-degenerate systems and a populated `support_freeze_wrong_over_base_h{1,5,10,20}`.

## How the results will be reported in the paper

A new appendix subsection of [`docs/neurips_sparse_koopman_multibasin.tex`](neurips_sparse_koopman_multibasin.tex) (working title: "Per-basin depth-slice robustness check") will report the same `|S|`, `H(B|S)`, `H(B|F)`, wrong-support-freeze h=1, h=20 metrics as Table 1, but on the per-basin deep slice. The denominators become `K/17` for the freeze diagnostic on these four systems (assuming the per-basin slice yields ≥ 2 templates per system, which is the expected outcome).

The current Table 1 (global slice) **does not change**: it remains the reference for the absolute-deep-states reading. The per-basin appendix becomes the reference for the "every basin contributes deep states" reading and is what supports the 17/17 wrong-support-freeze claim.

## Status

| Stage | State | Notes |
|---|---|---|
| Code changes (Python + sbatch wrappers) | done | Backward-compatible; `global` is still the default. |
| Launcher script | done | `scripts/queue_per_basin_deep_eval.sh`. |
| Markdown plan | done | this file. |
| Smoke test on 1 system / 1 seed | done | 16 s on a `long`-partition `salloc` node. `duffing_triple_well` returned `support_freeze_template_count=3` (vs `1` under global slice) and populated `wrong_over_base` ratios. |
| Submit per-basin shards (5 shards across 2 packets) | submitted 2026-04-27 | Job IDs: 9388212, 9388213 (LISTA packet) and 9388215, 9388216, 9388217 (boundary MLP packet); partition `long-cpu`, 16 GB, 12 h walltime, 4 CPUs. |
| Merge per-packet shard outputs | submitted with `afterok` dependency | Merge jobs 9388214 (LISTA) and 9388218 (boundary MLP). |
| Re-run `scripts/build_per_system_stats_and_forest.py` against new CSVs | pending | Add a per-basin variant that reads from `interpretability_per_basin_deep_pass1/`. |
| Add appendix table to LaTeX | pending | Will *not* overwrite Table 1; will be a new appendix table. |

### Job-ID provenance

The two `queue_manifest.json` files written during launch record the exact shard / merge job ID list:

- `results/transition_rich_basin_partition_final_seed10_20260409/interpretability_per_basin_deep_pass1/queue_manifest.json`
- `results/transition_rich_hardinit_mlp_controls_seed10_20260416/interpretability_per_basin_deep_pass1/queue_manifest.json`

Per-shard logs under `…/interpretability_per_basin_deep_pass1/logs/`.

## Linked documents

- Main paper draft: [`neurips_sparse_koopman_multibasin.tex`](neurips_sparse_koopman_multibasin.tex)
- Experiment log: [`EXPERIMENTS.md`](EXPERIMENTS.md)
- Paper-track status: [`PAPER_TRACK_STATUS.md`](PAPER_TRACK_STATUS.md)
- Stats build script: [`../scripts/build_per_system_stats_and_forest.py`](../scripts/build_per_system_stats_and_forest.py)
- Interpretability reducer: [`../tools/reduce_transition_rich_interpretability_metrics.py`](../tools/reduce_transition_rich_interpretability_metrics.py)
