# Subagent D Status: More Seeds On Headline Positives

Date: March 9, 2026

## Objective

Tighten uncertainty on the exact positive results most likely to appear in the paper main text, while keeping the promoted training recipes fixed:
- lower-dimensional encoder comparison: matched `200k` `generic_sparse_ns200k_best` vs promoted dense LISTA on the `21` low-dimensional systems
- Kuramoto `N=16`, `dt=0.00625`, `200k`: matched three-way comparison (`generic_sparse`, dense LISTA, `lista_blockdiag`)
- Kuramoto `N=32`, `dt=0.00625`, `200k`: matched confirmation (`generic_sparse`, `lista_blockdiag`)

## Current Evidence Read

- `docs/planning/paper_parallel_workstreams_20260309.md`
- `docs/PAPER_TRACK_STATUS.md`
- `docs/notes.tex`
- `results/paper_followup_recipes_200k_20260309/collect/forecasting_rows.csv`
- `results/dense_lista_paper_rerun_stage4_20260309/collect/forecasting_rows.csv`
- `results/kuramoto_dt00625_200k_compare_20260308/collect/forecasting_rows.csv`
- `results/kuramoto_n32_dt00625_200k_confirm_20260309/forecasting_rows.csv`
- `scripts/queue_paper_followup_recipes.sh`
- `scripts/queue_dense_lista_paper_rerun.sh`
- `scripts/queue_kuramoto_dt00625_200k_compare.sh`
- `scripts/run_paper_benchmark_array.sh`
- `scripts/collect_paper_benchmark.sh`
- `scripts/compare_paper_benchmark.sh`
- `tools/build_paper_followup_recipe_tasks.py`
- `tools/build_dense_lista_recipe_tasks.py`
- `tools/build_kuramoto_dimension_sweep_tasks.py`

Current read:
- Low-dimensional main-text claim in `docs/notes.tex` is the matched `200k` encoder comparison on the `21` low-dimensional systems, not the older `v4` anchor. Both matched roots currently have only seeds `0,1,2`.
- Current low-dimensional matched roots:
  - `generic_sparse_ns200k_best`: `21` systems, seeds `0,1,2`
  - `lista_dense_promoted_stage4`: `21` systems, seeds `0,1,2`
- Current Kuramoto `N=16` comparison already has `5` seeds and is unusually tight for `lista_blockdiag`:
  - `generic_sparse`: median `27.02`, std `5.58`
  - dense LISTA: median `13.84`, std `11.24`
  - `lista_blockdiag`: median `6.98`, std `0.079`, min/max `6.89/7.13`
- Current Kuramoto `N=32` confirmation has only `3` seeds:
  - `generic_sparse`: median `6.65`
  - `lista_blockdiag`: median `6.00`

Fairness note:
- Extending only dense LISTA on the low-dimensional headline set would not tighten the actual matched headline claim. The fair extension is matched new seeds for both `generic_sparse_ns200k_best` and promoted dense LISTA.

## Concrete Plan Before Queueing

Objective and claim being tested:
- Test whether the current headline positives remain stable when seed counts are extended with the same promoted recipes and no other recipe changes.

Primary baselines and fairness controls:
- Low-dimensional encoder claim:
  - matched `200k` MLP baseline: `generic_sparse_ns200k_best`
  - matched `200k` dense LISTA baseline: promoted Stage-4 dense root `lista_dense_promoted_stage4`
  - keep the pass-2 `dt` table fixed
  - keep all optimizer, sparsity, and architecture settings identical to the promoted roots
- Kuramoto `N=16`:
  - keep `dt=0.00625`, `200k` steps, and the existing three-way comparison exactly fixed
- Kuramoto `N=32`:
  - keep `dt=0.00625`, `200k` steps, and the existing two-way confirmation exactly fixed

Exact systems, seeds, horizons, and metrics:
- Low-dimensional systems (`21`): `duffing`, `lotka_volterra`, `blended`, `multiwell_gradient`, `multiwell_rotational`, `multiwell_energy`, `multiwell_strong_transition`, `dysts:Dadras`, `dysts:Duffing`, `dysts:QiChen`, `dysts:Sakarya`, `dysts:SprottTorus`, `dysts:Chua`, `dysts:MultiChua`, `dysts:DequanLi`, `dysts:LuChenCheng`, `dysts:SanUmSrisuchinwong`, `dysts:WangSun`, `dysts:ShimizuMorioka`, `dysts:RikitakeDynamo`, `dysts:Hadley`
- Low-dimensional additional seeds:
  - `generic_sparse_ns200k_best`: seeds `3,4`
  - promoted dense LISTA: seeds `3,4`
- Kuramoto `N=16` additional seeds:
  - seeds `5,6` for `generic_sparse`, dense LISTA, and `lista_blockdiag`
- Kuramoto `N=32` additional seeds:
  - seeds `3,4` for `generic_sparse` and `lista_blockdiag`
- Primary metric:
  - `H1000 best-periodic`
- Additional reporting targets:
  - low-dimensional `21`-system median `H1000 best-periodic` per model
  - low-dimensional dense-vs-generic win count on the `21` system medians
  - good-system counts (`H1000 best-periodic < 10`)
  - Kuramoto seed-median `H1000 best-periodic`, min/median/max, std, and all-seeds-good status

New target confidence summary:
- Low-dimensional main-text table should move from `3` to `5` matched seeds and report:
  - model-wise `21`-system median `H1000 best-periodic`
  - dense-vs-generic `21`-system head-to-head wins
  - low-dimensional good-system counts
  - seed-wise range over the `21`-system medians for both models
- Kuramoto `N=16` should move from `5` to `7` seeds and report:
  - seed-median `H1000 best-periodic`
  - min/median/max and std across seeds
  - all-seeds-good flag
- Kuramoto `N=32` should move from `3` to `5` seeds and report:
  - seed-median `H1000 best-periodic`
  - min/median/max and std across seeds
  - all-seeds-good flag

Acceptance criteria:
- Low-dimensional headline stays paper-usable if dense LISTA still beats matched `generic_sparse_ns200k_best` on the `21`-system median and remains in-band on all `21/21` systems.
- Kuramoto `N=16` stays paper-usable if `lista_blockdiag` remains inside the good band on the `7`-seed median and the added seeds do not introduce any bad or catastrophic seed.
- Kuramoto `N=32` stays paper-usable if `lista_blockdiag` remains inside the good band on the `5`-seed median and still stays below matched `generic_sparse`.

Failure criteria:
- Low-dimensional claim fails or weakens materially if dense LISTA loses the matched `21`-system median or drops out of band on any additional low-dimensional system.
- Kuramoto `N=16` weakens if any newly added block-diagonal seed lands outside the good band.
- Kuramoto `N=32` weakens if the added seeds erase the block-diagonal edge or push the `5`-seed median out of band.

Output roots and artifact names:
- Existing scratch roots to extend:
  - `/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/paper_followup_recipes/generic_sparse_ns200k_best`
  - `/network/scratch/l/lia/skae/dense_lista_paper_rerun_stage4_20260309/paper_rerun/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3`
  - `/network/scratch/l/lia/skae/kuramoto_dt00625_200k_compare_20260308/kuramoto_dt00625_200k/{generic_sparse,lista_dense,lista_blockdiag}`
  - `/network/scratch/l/lia/skae/kuramoto_n32_dt00625_200k_confirm_20260309/{generic_sparse,lista_blockdiag}`
- New metadata / collection roots:
  - `results/paper_parallel_20260309_d_lowdim_headline_more_seeds/`
  - `results/paper_parallel_20260309_d_kuramoto_n16_more_seeds/`
  - `results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/`

## Local QA Before Queueing

Planned local QA checklist:
- Build the low-dimensional generic task table locally with `uv run python tools/build_paper_followup_recipe_tasks.py ...` and confirm:
  - exactly `42` tasks (`21` systems x `2` seeds x `1` recipe)
  - only the `21` low-dimensional systems are present
  - seeds are exactly `3,4`
- Build the low-dimensional dense task table locally with `uv run python tools/build_dense_lista_recipe_tasks.py ...` and confirm:
  - exactly `42` tasks
  - only the `21` low-dimensional systems are present
  - seeds are exactly `3,4`
- Build the Kuramoto `N=16` task table locally and confirm:
  - exactly `6` tasks (`3` models x `2` seeds)
  - seeds are exactly `5,6`
  - root specs point at the existing `kuramoto_dt00625_200k_compare_20260308` scratch roots
- Build or expand the Kuramoto `N=32` launch commands locally and confirm:
  - exactly `4` tasks (`2` models x `2` seeds)
  - seeds are exactly `3,4`
  - `--log_dir` targets the existing `kuramoto_n32_dt00625_200k_confirm_20260309` model roots
- Confirm all queue paths avoid the prior `sbatch --export` comma-CSV bug pattern.
- Confirm task counts with `wc -l`, inspect root-spec files, and sample task rows before submission.
- Check cluster availability with `savail` before queueing if the command is available.

QA performed:
- `savail` check succeeded before submission:
  - `a100 2/32`
  - `l40s 2/352`
  - `rtx8000 49/376`
  - `v100 11/56`
- Built lightweight task artifacts by reusing the existing canonical task TSVs and filtering / reseeding them locally:
  - `results/paper_parallel_20260309_d_lowdim_headline_more_seeds/task_tables/generic_sparse_lowdim_seeds_3_4.tsv`
  - `results/paper_parallel_20260309_d_lowdim_headline_more_seeds/task_tables/dense_lowdim_seeds_3_4.tsv`
  - `results/paper_parallel_20260309_d_kuramoto_n16_more_seeds/task_tables/kuramoto_n16_seeds_5_6.tsv`
  - `results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/task_tables/kuramoto_n32_seeds_3_4.tsv`
- Confirmed row counts with `wc -l`:
  - low-dimensional generic: `43` lines => `42` tasks
  - low-dimensional dense: `43` lines => `42` tasks
  - Kuramoto `N=16`: `7` lines => `6` tasks
  - Kuramoto `N=32`: `5` lines => `4` direct-launch tasks
- Confirmed exact seed sets with local CSV inspection:
  - low-dimensional generic: seeds `3,4`
  - low-dimensional dense: seeds `3,4`
  - Kuramoto `N=16`: seeds `5,6`
  - Kuramoto `N=32`: seeds `3,4`
- Confirmed exact system coverage:
  - low-dimensional generic and dense task tables contain exactly `21` systems
  - Kuramoto `N=16` contains exactly the three intended model variants
  - Kuramoto `N=32` contains exactly `generic_sparse` and `lista_blockdiag`
- Confirmed root-spec files:
  - `results/paper_parallel_20260309_d_lowdim_headline_more_seeds/root_specs/lowdim_headline_roots.txt`
  - `results/paper_parallel_20260309_d_kuramoto_n16_more_seeds/root_specs/kuramoto_n16_roots.txt`
  - `results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/root_specs/kuramoto_n32_roots.txt`
- Expanded and inspected the exact `uv run python tools/train.py ...` launch commands for the four Kuramoto `N=32` jobs before submission.
- All submissions avoided `sbatch --export`; array jobs used the existing TSV-driven runner, and the four `N=32` jobs used explicit `sbatch --wrap` commands.
- Post-submit `squeue` check confirmed every train job plus the dependent collect/compare jobs were present with the expected dependency states.

## What Was Queued

Final local submission from this subagent:
- Use the job IDs in this subsection as the current source of truth. An earlier provisional note with `8914715-8914717` was already in this file when this update was written; the final queued set from this pass is below.

Low-dimensional matched encoder extension:
- Task artifacts:
  - `results/paper_parallel_20260309_d_lowdim_headline_more_seeds/task_tables/generic_sparse_lowdim_seeds_3_4.tsv`
  - `results/paper_parallel_20260309_d_lowdim_headline_more_seeds/task_tables/dense_lowdim_seeds_3_4.tsv`
  - `results/paper_parallel_20260309_d_lowdim_headline_more_seeds/root_specs/lowdim_headline_roots.txt`
- Train arrays:
  - `TASK_TSV=results/paper_parallel_20260309_d_lowdim_headline_more_seeds/task_tables/generic_sparse_lowdim_seeds_3_4.tsv BASE_OUT=/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309 sbatch --array=0-41 scripts/run_paper_benchmark_array.sh`
    - job ID: `8914743`
  - `TASK_TSV=results/paper_parallel_20260309_d_lowdim_headline_more_seeds/task_tables/dense_lowdim_seeds_3_4.tsv BASE_OUT=/network/scratch/l/lia/skae/dense_lista_paper_rerun_stage4_20260309 sbatch --array=0-41 scripts/run_paper_benchmark_array.sh`
    - job ID: `8914744`
- Dependent collection / compare:
  - `ROOT_SPECS_FILE=results/paper_parallel_20260309_d_lowdim_headline_more_seeds/root_specs/lowdim_headline_roots.txt OUT_DIR=results/paper_parallel_20260309_d_lowdim_headline_more_seeds/collect PAPER_SUMMARY=1 sbatch --dependency=afterany:8914743:8914744 scripts/collect_paper_benchmark.sh`
    - job ID: `8914745`
  - `ROWS_CSV=results/paper_parallel_20260309_d_lowdim_headline_more_seeds/collect/forecasting_rows.csv OUT_DIR=results/paper_parallel_20260309_d_lowdim_headline_more_seeds/compare CANDIDATE_ROOTS_CSV=lista_dense_promoted_stage4 ANCHOR_ROOT=generic_sparse_ns200k_best HORIZON=1000 sbatch --dependency=afterany:8914745 scripts/compare_paper_benchmark.sh`
    - job ID: `8914746`

Kuramoto `N=16` matched extension:
- Task artifacts:
  - `results/paper_parallel_20260309_d_kuramoto_n16_more_seeds/task_tables/kuramoto_n16_seeds_5_6.tsv`
  - `results/paper_parallel_20260309_d_kuramoto_n16_more_seeds/root_specs/kuramoto_n16_roots.txt`
- Train array:
  - `TASK_TSV=results/paper_parallel_20260309_d_kuramoto_n16_more_seeds/task_tables/kuramoto_n16_seeds_5_6.tsv BASE_OUT=/network/scratch/l/lia/skae/kuramoto_dt00625_200k_compare_20260308 sbatch --array=0-5 scripts/run_paper_benchmark_array.sh`
    - job ID: `8914748`
- Dependent collection / compare:
  - `ROOT_SPECS_FILE=results/paper_parallel_20260309_d_kuramoto_n16_more_seeds/root_specs/kuramoto_n16_roots.txt OUT_DIR=results/paper_parallel_20260309_d_kuramoto_n16_more_seeds/collect PAPER_SUMMARY=1 sbatch --dependency=afterany:8914748 scripts/collect_paper_benchmark.sh`
    - job ID: `8914749`
  - `ROWS_CSV=results/paper_parallel_20260309_d_kuramoto_n16_more_seeds/collect/forecasting_rows.csv OUT_DIR=results/paper_parallel_20260309_d_kuramoto_n16_more_seeds/compare CANDIDATE_ROOTS_CSV=lista_dense,lista_blockdiag ANCHOR_ROOT=generic_sparse HORIZON=1000 sbatch --dependency=afterany:8914749 scripts/compare_paper_benchmark.sh`
    - job ID: `8914750`

Kuramoto `N=32` matched extension:
- Task artifacts:
  - `results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/task_tables/kuramoto_n32_seeds_3_4.tsv`
  - `results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/root_specs/kuramoto_n32_roots.txt`
- Direct train jobs (all via `sbatch --wrap` running `uv run python tools/train.py ...` against the existing confirmation roots):
  - `generic_sparse`, seed `3`
    - job ID: `8914751`
  - `generic_sparse`, seed `4`
    - job ID: `8914752`
  - `lista_blockdiag`, seed `3`
    - job ID: `8914753`
  - `lista_blockdiag`, seed `4`
    - job ID: `8914754`
- Dependent collection / compare:
  - `ROOT_SPECS_FILE=results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/root_specs/kuramoto_n32_roots.txt OUT_DIR=results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/collect PAPER_SUMMARY=1 sbatch --dependency=afterany:8914751:8914752:8914753:8914754 scripts/collect_paper_benchmark.sh`
    - job ID: `8914755`
  - `ROWS_CSV=results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/collect/forecasting_rows.csv OUT_DIR=results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/compare CANDIDATE_ROOTS_CSV=lista_blockdiag ANCHOR_ROOT=generic_sparse HORIZON=1000 sbatch --dependency=afterany:8914755 scripts/compare_paper_benchmark.sh`
    - job ID: `8914756`

- Coordinator-side queue start to unblock the paper workstream while the subagent was still finishing local builder QA.
- Built lightweight derived task tables directly from the existing experiment task TSVs, preserving the agreed recipes and system lists:
  - `results/paper_parallel_20260309_d_lowdim_headline_more_seeds/task_tables/generic_sparse_lowdim_seeds_3_4.tsv`
  - `results/paper_parallel_20260309_d_lowdim_headline_more_seeds/task_tables/dense_lowdim_seeds_3_4.tsv`
  - `results/paper_parallel_20260309_d_kuramoto_n16_more_seeds/task_tables/kuramoto_n16_seeds_5_6.tsv`
- Verified task counts before submission:
  - low-dimensional `generic_sparse_ns200k_best`: `42` tasks
  - low-dimensional promoted dense LISTA: `42` tasks
  - Kuramoto `N=16`: `6` tasks
- Submitted SLURM arrays:
  - low-dimensional promoted dense LISTA seeds `3,4` into `/network/scratch/l/lia/skae/dense_lista_paper_rerun_stage4_20260309`:
    - `sbatch --array=0-41 scripts/run_paper_benchmark_array.sh`
    - job ID: `8914715`
  - low-dimensional `generic_sparse_ns200k_best` seeds `3,4` into `/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309`:
    - `sbatch --array=0-41 scripts/run_paper_benchmark_array.sh`
    - job ID: `8914716`
  - Kuramoto `N=16` matched three-way seeds `5,6` into `/network/scratch/l/lia/skae/kuramoto_dt00625_200k_compare_20260308`:
    - `sbatch --array=0-5 scripts/run_paper_benchmark_array.sh`
    - job ID: `8914717`
- Queue verification:
  - `squeue -u lia` showed `8914715`, `8914716`, and `8914717` present in partition `long` in `PD` state immediately after submission.
- Superseded note from the earlier provisional block:
  - Kuramoto `N=32` seeds `3,4` were subsequently queued in the final local submission above under train jobs `8914751-8914754`.

## Results / Observations

- The final queued set covers all three intended headline extensions:
  - low-dimensional matched encoder comparison to `5` seeds
  - Kuramoto `N=16` to `7` seeds
  - Kuramoto `N=32` to `5` seeds
- The existing builder scripts were not the fastest reliable local QA path in this environment because importing `skae` pulled the heavier training stack. Reusing the already materialized canonical task TSVs produced the same task content without changing any code or training recipe.
- `squeue` after submission showed:
  - `8914743_[0-41]`, `8914744_[0-41]`, `8914748_[0-5]`, `8914751-8914754` pending on priority
  - `8914745`, `8914746`, `8914749`, `8914750`, `8914755`, `8914756` pending on dependency

## Proposed Updates To Global Status Docs

- After the collectors finish, update the global docs to say the paper-facing positive claims now have the following queued seed extensions in flight:
  - low-dimensional matched encoder comparison: `3 -> 5` matched seeds
  - Kuramoto `N=16`: `5 -> 7` seeds
  - Kuramoto `N=32`: `3 -> 5` seeds
- If the new seeds preserve the current rankings, update the main-text evidence summary to quote the widened seed counts and the refreshed confidence summaries from:
  - `results/paper_parallel_20260309_d_lowdim_headline_more_seeds/collect`
  - `results/paper_parallel_20260309_d_kuramoto_n16_more_seeds/collect`
  - `results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/collect`
- If the new seeds materially weaken any claim, use these three collection directories as the first source of truth for revising the paper narrative.
