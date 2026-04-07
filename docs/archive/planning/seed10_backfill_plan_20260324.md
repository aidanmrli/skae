# Seed-10 Backfill Plan (2026-03-24)

## Goal

Raise every table-facing system/root setting cited in `docs/review_main_results_tables_20260314.tex` to at least seeds `0-9`, then refresh the paper-facing collectors and mechanism summaries that currently read those runs.

## Audit Artifacts

- Backfill audit CSV: `results/seed10_backfill_20260324/task_tables/seed10_backfill_audit.csv`
- Backfill task TSV: `results/seed10_backfill_20260324/task_tables/seed10_backfill_tasks.tsv`
- Backfill summary JSON: `results/seed10_backfill_20260324/task_tables/seed10_backfill_summary.json`

## Scope

The audit covers:

- Main 29-system benchmark roots from `results/paper_zero_sparse_benchmark_200k_20260321/collect/forecasting_rows.csv`
- Hard-system roots from `results/zero_sparse_hard_systems_20260321/collect/...`
- Repaired Kuramoto dimension block-diagonal MLP retry roots collected into `results/seed10_backfill_20260324/generated/kuramoto_dimension_blockdiag_retry1_collect/`
- Hopfield `N=64` table roots collected directly into `results/seed10_backfill_20260324/generated/hopfield_n64_table_collect/`
- Canonical Kuramoto `dt=0.01` roots from `results/kuramoto_dt0p01_200k_canonical_20260323/collect/forecasting_rows.csv`

Notable exclusions:

- Corrected 4-basin CLV training roots already exceed the target with 15 seeds, so no new training jobs are needed there.
- The Hopfield `N=64` table is refreshed from direct table-root collectors rather than the mixed old `hopfield_n64` collector, which had stale cross-pattern row selection.

## Missing-Seed Summary

Global totals from the audit:

- Table-facing root/system groups audited: `238`
- Missing training runs needed to reach seeds `0-9`: `1412`

Source-level missing-run counts:

- Main benchmark: `1092`
- Kuramoto `N=16`, identical frequencies: `19`
- Kuramoto `N=16`, uniform spread: `25`
- Kuramoto dimension sweep (existing sparse / zero / LISTA roots): `100`
- Kuramoto dimension sweep repaired block-diagonal MLP retry roots: `25`
- Hopfield `N=16`: `21`
- Hopfield `N=64` table roots: `35`
- Fixed 8-basin CLV, `dt=0.005`: `35`
- Fixed 8-basin CLV, `dt=0.0025`: `35`
- Canonical Kuramoto `dt=0.01`: `25`
- Corrected 4-basin CLV: `0`

Main benchmark interpretation:

- `generic_sparse_ns200k_best`, `generic_sparse_sc0_ns200k_best`, and `lista_dense_promoted_stage4` currently have seeds `0-4` on 21 systems and only `0-2` on the 8 higher-cost systems, so they need `161` new runs each for the benchmark tables.
- `lista_blockdiag_ns200k_denseopt_sc3em3` and `lista_blockdiag_ns200k_denseopt_sc6em3` currently have only seeds `0-2` on all 29 systems, so they need `203` new runs each.

Hard-system interpretation:

- Kuramoto `N=16` identical: sparse / dense / blockdiag-LISTA already have 7 seeds, while zero-sparse MLP and repaired block-diagonal MLP have 5; all are topped up to 10.
- Kuramoto dimension sweep roots are uniformly at 5 seeds and need seeds `5-9`, including the repaired block-diagonal MLP retry roots.
- Hopfield `N=16`, Hopfield `N=64`, and both fixed 8-basin CLV settings are still at 3 seeds and need seeds `3-9`.
- Canonical Kuramoto `dt=0.01` is uniformly at 5 seeds and needs seeds `5-9`.

## Queue Plan

Training:

1. Submit one global explicit-logdir SLURM array using `scripts/run_explicit_backfill_array.sh` and `results/seed10_backfill_20260324/task_tables/seed10_backfill_tasks.tsv`.
2. New seeds are written directly beside the existing runs in the original scratch roots, including the custom CLV and repaired-fairness layouts.

Post-training refresh:

1. Re-run the main benchmark collector, comparisons, and fixed-cadence summary using the existing benchmark root-spec file.
2. Re-run hard-system collectors and comparisons for:
   - Kuramoto `N=16` identical
   - Kuramoto `N=16` uniform spread
   - Kuramoto dimension sweep using `results/seed10_backfill_20260324/generated/kuramoto_dimension_table_roots.txt`
   - Hopfield `N=16`
   - Hopfield `N=64` using `results/seed10_backfill_20260324/generated/hopfield_n64_table_roots.txt`
   - fixed 8-basin CLV at both cited `dt`
   - canonical Kuramoto `dt=0.01`
3. Re-run the zero-sparsity mechanism family with seed overrides:
   - support-alignment benchmark on seeds `0-9`
   - label-free clustering on benchmark seeds `0-9` and Kuramoto seeds `0-9`
   - direct Kuramoto support audit on seeds `0-9`
   - recurring-support local-linearity on benchmark / Kuramoto / CLV seeds `0-9`
   - corrected 4-basin CLV representation follow-up on seeds `0-9` with parity roots disabled

## Queue Record

Current queue state on 2026-03-24:

- Initial chunked submission succeeded for `0-799` as eight `100`-task arrays:
  - `9036948` -> `0-99`
  - `9036949` -> `100-199`
  - `9036950` -> `200-299`
  - `9036951` -> `300-399`
  - `9036953` -> `400-499`
  - `9036954` -> `500-599`
  - `9036955` -> `600-699`
  - `9036956` -> `700-799`
- A direct ninth chunk for `800-899` was accepted as `9037646`.
- After inspection, only `9036948` had started running; every later array was still fully pending. To reduce submit-count pressure without disturbing active work, the fully pending arrays were canceled:
  - `9036949`
  - `9036950`
  - `9036951`
  - `9036953`
  - `9036954`
  - `9036955`
  - `9036956`
  - `9037646`
- Replacement submission:
  - `ARRAY_OFFSET=100 TASK_TSV=results/seed10_backfill_20260324/task_tables/seed10_backfill_tasks.tsv sbatch --parsable --array=0-699 scripts/run_explicit_backfill_array.sh`
  - accepted as `9037651`, covering task rows `100-799`
- The remaining tail still hit `AssocMaxSubmitJobLimit` when submitted directly, so a long-CPU launcher job was added:
  - `sbatch --parsable scripts/queue_seed10_backfill_tail_and_refresh_20260324.sh`
  - accepted as `9037719`
  - this launcher will retry submission of the tail array for task rows `800-1411`, then queue all dependent benchmark / hard-system collectors, comparisons, fixed-cadence summary, and the zero-sparsity mechanism wrapper

Active training queue layout now:

- `9036948` -> task rows `0-99`
- `9037651` -> task rows `100-799`
- `9039307` -> task rows `800-1411`

Launcher result:

- launcher `9037719` completed the delayed submission and wrote `results/seed10_backfill_20260324/queue_record.json`
- dependency barrier: `9039308`
- benchmark refresh:
  - collect `9039309`
  - compare vs sparse `9039310`
  - compare vs zero-sparse `9039346`
  - compare vs dense `9039347`
  - fixed cadence `9039348`
- hard-system refresh:
  - Kuramoto `N=16` identical collect/compare: `9039349` / `9039350`
  - Kuramoto uniform-spread collect/compare: `9039351` / `9039352`
  - Kuramoto dimension collect/compare: `9039353` / `9039369`
  - Hopfield `N=16` collect/compare: `9039370` / `9039371`
  - Hopfield `N=64` collect/compare: `9039372` / `9039373`
  - CLV `8`-basin `dt=0.005` collect/compare: `9039374` / `9039375`
  - CLV `8`-basin `dt=0.0025` collect/compare: `9039402` / `9039403`
  - canonical Kuramoto `dt=0.01` collect/compare: `9039404` / `9039405`
- zero-sparsity mechanism queue wrapper: `9039406`

Post-refresh verification on 2026-03-26:

- `squeue` is empty and the full March 25 bulk chain is complete: tail array `9039307`, barrier `9039308`, refresh jobs `9039309-9039406`, and mechanism subjobs `9059785-9059790` all finished `0:0`
- Refreshed collector artifacts landed on March 25 under the benchmark, hard-system, canonical Kuramoto, and mechanism result roots
- Verification rerun [results/seed10_backfill_20260324/verification_refresh/seed10_backfill_summary.json](/home/mila/l/lia/skae/results/seed10_backfill_20260324/verification_refresh/seed10_backfill_summary.json) found one residual missing seed instead of zero:
  - source: `results/zero_sparse_hard_systems_20260321/collect/kuramoto_dimension/forecasting_rows.csv`
  - root: `generic_sparse_sc0_n8`
  - missing seed: `6`
- Residual repair chain is now queued:
  - training backfill `9074821`
  - recollect `9074822`
  - recompare `9074823`

Custom root-spec files used by the refresh stage:

- `results/seed10_backfill_20260324/generated/kuramoto_dimension_table_roots.txt`
- `results/seed10_backfill_20260324/generated/hopfield_n64_table_roots.txt`

Notes:

- `results/seed10_backfill_20260324/array_jobs.txt` now reflects the three active training arrays: `9036948`, `9037651`, and `9039307`.
- Corrected 4-basin CLV already had 15 seeds, so no new training jobs were queued there.
