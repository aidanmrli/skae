# Subagent G Status: Kuramoto Checkpoint Selection

Date: March 9, 2026

## Objective

- Quantify how much the current Kuramoto paper-facing conclusions change when checkpoint selection uses `evaluation_results_best.json` (validation-selected) versus `evaluation_results_last.json` (last checkpoint), with forecasting-first emphasis at `H100`, `H500`, and `H1000`.
- Decide whether the paper should keep `best` as the official rule, switch to `last`, or present `last` only as a diagnostic limitation / ablation.

## Current Evidence Read

- `docs/planning/paper_parallel_workstreams_20260309.md`
- `docs/EXPERIMENTS.md`
- `docs/PAPER_TRACK_STATUS.md`
- `tools/collect_forecasting_roots.py`
- `results/kuramoto_dt00625_200k_compare_20260308/root_specs/kuramoto_dt00625_200k_roots.txt`
- Scratch run roots confirmed:
  - `/network/scratch/l/lia/skae/intrinsic_hd_dt_rescue_20260308_rerun1`
  - `/network/scratch/l/lia/skae/kuramoto_dt00625_200k_compare_20260308`
- Existing docs already record a `best` vs `last` mismatch on the older focused intrinsic-HD Kuramoto rerun, but not yet a completed ablation on the newer `dt=0.00625`, `200k` Kuramoto comparison.

## Concrete Plan Before Queueing

- Objective and claim being tested:
  - Test whether the Kuramoto hard-system story depends materially on validation-selected checkpoints.
  - Primary claim under audit: `lista_blockdiag` is the strongest Kuramoto encoder under the current `dt=0.00625`, `200k` setting, and smaller `dt` plus longer training rescue Kuramoto under periodic reencoding.
- Primary baselines and fairness controls:
  - Use identical run roots for `best` and `last`; only the evaluation artifact name changes.
  - Analyze two Kuramoto-focused roots named in the workstream:
    - focused intrinsic-HD rerun: `/network/scratch/l/lia/skae/intrinsic_hd_dt_rescue_20260308_rerun1`
    - focused Kuramoto `dt=0.00625`, `200k` comparison: `/network/scratch/l/lia/skae/kuramoto_dt00625_200k_compare_20260308/kuramoto_dt00625_200k`
  - Restrict the analysis to Kuramoto rows only.
  - Compare matched seeds and matched arms wherever possible.
- Exact systems, seeds, horizons, and metrics:
  - Systems: Kuramoto only.
  - Older focused rerun:
    - models: `generic_sparse`, `lista_blockdiag`
    - `dt in {0.025, 0.0125}`
    - all available seeds per arm (`0,1,2`)
    - compare `H1000` first, then confirm `H100` and `H500` if needed.
  - Newer focused Kuramoto comparison:
    - models: `generic_sparse`, `lista_dense`, `lista_blockdiag`
    - `dt=0.00625`
    - seeds `0..4`
    - horizons `100`, `500`, `1000`
  - Metrics:
    - per-seed and seed-median `best_periodic_mean`
    - good-band counts at `H1000 < 10`
    - pairwise `last - best` and ratio `last / best` for matched runs
    - rank changes between models under each checkpoint rule
- Acceptance criteria:
  - No new queue is needed if existing `evaluation_results_best.json` and `evaluation_results_last.json` cover all targeted Kuramoto runs and local recollection reproduces the expected row counts.
  - A revised paper rule is only justified if `last` changes the headline model ranking or materially changes the qualitative claim (for example, a previously non-good model becomes robustly good, or `best` clearly understates the winning model by a large margin across seeds).
- Failure criteria:
  - Missing `evaluation_results_last.json` files for a nontrivial fraction of the targeted Kuramoto runs.
  - Collector mismatch or row-count mismatch that prevents a fair matched comparison.
  - Any required recollection command failing locally.
- Output root and artifact names:
  - Prefer no persistent repo outputs.
  - If local recollection is needed, use temporary directories under `/tmp/paper_parallel_20260309_g_kuramoto_checkpoint_selection/` with subdirs:
    - `best_collect/`
    - `last_collect/`
    - `analysis/`
  - If a queued recollection becomes necessary, use a unique scratch root prefixed `paper_parallel_20260309_g_...`.

## Local QA Before Queueing

- Verify the exact scratch roots and confirm both `evaluation_results_best.json` and `evaluation_results_last.json` exist for the targeted Kuramoto runs.
- Smoke-test `tools/collect_forecasting_roots.py` locally with `uv run` on the Kuramoto-focused roots before any `sbatch`.
- Confirm collected row counts match expectations:
  - older focused rerun: `24` Kuramoto rows per eval-file selection (`2` models x `2` dts x `2` sparsity settings x `3` seeds)
  - newer `dt=0.00625`, `200k` comparison: `15` Kuramoto rows per eval-file selection (`3` models x `5` seeds)
- Check that matched runs preserve the same `run_dir`, `seed`, `env_dt`, model, and arm metadata across `best` and `last`.
- Compute medians directly from collected rows and spot-check at least one known doc claim from the older focused rerun.
- Do not queue anything if offline recollection already answers the workstream.

Performed:
- Confirmed artifact coverage via direct file scan:
  - `pilot`: `24` `best` + `24` `last` Kuramoto eval files
  - `compare`: `15` `best` + `15` `last` Kuramoto eval files
- Attempted a local `uv run` collector smoke on a copied one-run tree under `/tmp/paper_parallel_20260309_g_kuramoto_checkpoint_selection_smoke_tree.KanmJo`, but the subagent worktree `.venv` was incomplete and `tools/collect_forecasting_roots.py` failed on import with `ModuleNotFoundError: typing_extensions`.
- Since no queue was needed and the evaluation JSONs already existed, completed the ablation with direct `find` + `jq` + `awk` aggregation into `/tmp/paper_parallel_20260309_g_kuramoto_checkpoint_selection_analysis.Q2UG1V/raw.tsv`.
- QA checks passed:
  - matched `best` / `last` rows share the same `run_dir`, `seed`, `dt`, model, and arm metadata
  - direct `best` medians on the newer Kuramoto comparison reproduce the existing reported values (`27.02`, `13.84`, `6.98` at `H1000`)
  - direct pilot aggregation reproduces the older documented checkpoint-mismatch claim (`23.40 -> 14.64` for `lista_blockdiag`, `dt=0.0125`, aggregated over the two sparsity arms)
  - the winning older pilot arm also reproduces the documented diagnostic (`14.36 -> 13.91` for `lista_blockdiag`, `dt=0.0125`, `sp=0.0005`)

## What Was Queued

- No queue submitted.
- SLURM job IDs: none.
- Local commands used:
  - `find /network/scratch/l/lia/skae/intrinsic_hd_dt_rescue_20260308_rerun1 -type f \( -name 'evaluation_results_best.json' -o -name 'evaluation_results_last.json' \)`
  - `find /network/scratch/l/lia/skae/kuramoto_dt00625_200k_compare_20260308/kuramoto_dt00625_200k -type f \( -name 'evaluation_results_best.json' -o -name 'evaluation_results_last.json' \)`
  - `jq -r '[input_filename, .kuramoto.best_periodic."100".mean, .kuramoto.best_periodic."500".mean, .kuramoto.best_periodic."1000".mean] | @tsv' ...`
  - `awk` summaries over `/tmp/paper_parallel_20260309_g_kuramoto_checkpoint_selection_analysis.Q2UG1V/raw.tsv` for row counts, medians, good-band counts, and paired `last-best` deltas.

## Results / Observations

- Conclusion: no new queue is needed. Existing Kuramoto checkpoints already answer the checkpoint-selection ablation.
- Older focused intrinsic-HD rerun (`pilot`) reproduces the previously documented mismatch:
  - `lista_blockdiag`, `dt=0.0125`, aggregated over both sparsity arms:
    - `best`: `H100/H500/H1000 = 0.9263 / 10.2522 / 23.3951`
    - `last`: `H100/H500/H1000 = 0.5575 / 7.1394 / 14.6408`
  - winning older pilot arm `lista_blockdiag`, `dt=0.0125`, `sp=0.0005`:
    - `best`: `0.5526 / 7.0905 / 14.3588`
    - `last`: `0.5360 / 6.9147 / 13.9055`
- Newer focused Kuramoto `dt=0.00625`, `200k` comparison (`compare`) is checkpoint-selection stable enough that the paper conclusion does not change:
  - `generic_sparse`:
    - `best`: `H100/H500/H1000 = 0.0343 / 1.4972 / 27.0193`
    - `last`: `0.0374 / 1.5280 / 29.4255`
  - `lista_dense`:
    - `best`: `0.2194 / 4.1222 / 13.8445`
    - `last`: `0.2280 / 4.6001 / 17.6263`
  - `lista_blockdiag`:
    - `best`: `0.1536 / 2.6834 / 6.9804`
    - `last`: `0.1546 / 2.6841 / 6.9972`
- Good-band counts on the newer comparison at `H1000 < 10`:
  - `lista_blockdiag`: `5/5` under both `best` and `last`
  - `generic_sparse`: `0/5` under both
  - `lista_dense`: `0/5` under both
- Paired median `H1000` delta (`last - best`) on the newer comparison:
  - `generic_sparse`: `+0.0559`
  - `lista_dense`: `+0.7707`
  - `lista_blockdiag`: `+0.0234`
- Interpretation:
  - the strong older-pilot mismatch is real, but it is tied to the earlier `dt=0.0125`, `20k` Kuramoto rescue grid rather than the current headline `dt=0.00625`, `200k` result
  - on the current paper-facing Kuramoto comparison, switching from `best` to `last` does not change the model ranking (`lista_blockdiag < lista_dense < generic_sparse`) and does not change which models are in the good-forecast band
  - `last` is not uniformly better even on Kuramoto; on the newer comparison its seed-median `H1000` is slightly worse for all three models
  - no revised paper model-selection rule is justified from this ablation; keep `evaluation_results_best.json` as the official paper metric and frame `evaluation_results_last.json` as an older Kuramoto diagnostic / limitation note

## Proposed Updates To Global Status Docs

- `docs/EXPERIMENTS.md`
  - add a short checkpoint-selection ablation note under the Kuramoto follow-up evidence:
    - the older `dt=0.0125`, `20k` pilot mismatch reproduces (`lista_blockdiag`, aggregated `dt=0.0125`: `H1000 23.40 -> 14.64`; winning arm `14.36 -> 13.91`)
    - the current `dt=0.00625`, `200k` Kuramoto comparison is selection-stable (`generic_sparse 27.02 -> 29.43`, `lista_dense 13.84 -> 17.63`, `lista_blockdiag 6.98 -> 7.00`)
    - ranking and good-band membership are unchanged, so `best` should remain the official paper rule
- `docs/PAPER_TRACK_STATUS.md`
  - add a high-level sentence that the checkpoint-selection ablation is now complete:
    - `evaluation_results_last.json` is still worth mentioning as an older Kuramoto diagnostic
    - it does not overturn the current headline Kuramoto claim or justify changing the official selection rule away from `best`
