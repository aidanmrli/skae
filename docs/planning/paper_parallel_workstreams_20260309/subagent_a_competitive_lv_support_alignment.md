# Subagent A Status: `competitive_lv` Support Alignment

Date: March 9, 2026

## Objective

Upgrade the basin-support evidence from low-dimensional systems to an intrinsic high-dimensional positive control by measuring support uniqueness, cosine basin separation, and latent clustering quality on `competitive_lv` for the current paper-facing headline roots.

## Current Evidence Read

- Coordination brief:
  - `docs/planning/paper_parallel_workstreams_20260309.md`
- Paper status / experiment context:
  - `docs/PAPER_TRACK_STATUS.md`
  - `docs/EXPERIMENTS.md`
  - `docs/high_dim_supervisor_brief_20260305.md`
- Existing result roots / task tables:
  - `results/paper_followup_recipes_200k_20260309/`
  - `results/dense_lista_paper_rerun_stage4_20260309/`
- Tooling audit:
  - `skae/data.py`
  - `skae/basin_utils.py`
  - `tools/evaluate_support_uniqueness.py`
  - `tools/evaluate_basin_structure.py`
  - `tools/evaluate_latent_basin_clustering.py`
- Key audit findings:
  - `skae.basin_utils.BasinLabeledDataset` already supports `competitive_lv` through environment-native `basin_label(state)` and names remapped labels as survivor masks.
  - `tools/evaluate_support_uniqueness.py` already uses `skae.basin_utils.BasinLabeledDataset`, so `competitive_lv` support/cosine evaluation is already wired.
  - `tools/evaluate_latent_basin_clustering.py` still uses an older local dataset implementation restricted to `duffing` / `lyapunov` / `dysts:Duffing`, so it does not currently run on `competitive_lv` without a minimal wrapper or patch.
  - `tools/evaluate_basin_structure.py` is specialized to `StructuredLISTAKM`; it is not the right path for the required three-way `generic_sparse` / dense LISTA / block-diagonal comparison.
  - The fair three-way root set is available entirely from existing checkpoints; no retraining is needed unless local offline analysis fails.
  - For `competitive_lv`, the best available block-diagonal comparator among the paper-followup `200k` roots is `lista_blockdiag_ns200k_denseopt_sc3em3`:
    - `sc=0.003`: seed `H1000` best-periodic `{3.068e+04, 0.2575, 0.3544}`, median `0.3544`
    - `sc=0.006`: seed `H1000` best-periodic `{1.3560, 0.8260, 5.404e+11}`, median `1.3560`

## Concrete Plan Before Queueing

- Claim being tested:
  - `competitive_lv` should provide intrinsic-HD positive-control evidence that paper-facing models learn basin-discriminative latent supports, even when forecasting quality differs across architectures.
- Primary comparison:
  - `generic_sparse_ns200k_best`
  - `lista_dense_promoted_stage4`
  - `lista_blockdiag_ns200k_denseopt_sc3em3`
- Checkpoint policy:
  - Use existing validation-selected `checkpoint.pt` files, not `last.pt`, to stay aligned with the official paper-facing best-checkpoint collection.
- Exact runs:
  - System: `competitive_lv`
  - Seeds: `0, 1, 2` for every root
  - Eval dataset seed: `42`
  - Trajectories: start with a local smoke test on `12`; if clean, run the full offline pass on `100`
  - Trajectory length: `500`
  - Long-rollout steps for basin identification: `5000`
  - Device: `cpu`
- Metrics to report:
  - Support uniqueness:
    - `mode_uniqueness_rate`
    - `mean_basin_consistency`
    - `mean_pairwise_jaccard`
    - `unique_mode_supports / num_basins`
  - Cosine separation:
    - `mean_intra_basin_cosine`
    - `mean_inter_basin_cosine`
    - `cosine_separation_score`
  - Clustering / separability:
    - `silhouette_score`
    - `adjusted_rand_index`
    - `kmeans_purity`
    - `linear_classifier_accuracy`
- Fairness controls:
  - Same `competitive_lv` environment settings from the saved configs
  - Same evaluation seed and trajectory generator across all models
  - Same checkpoint type (`checkpoint.pt`) across all models
  - Same trajectory count / rollout settings across all models
- Output root and artifact names:
  - Output root: `results/paper_parallel_20260309_a_competitive_lv_support_alignment/`
  - Minimal helper script if needed: `tools/paper_parallel_20260309_a_competitive_lv_support_alignment.py`
  - Expected artifacts:
    - per-run JSON under `per_run/<root_label>/seed_<n>/`
    - aggregate `competitive_lv_support_alignment_summary.json`
    - aggregate `competitive_lv_support_alignment_summary.md`
- Acceptance criteria:
  - At least one paper-facing root shows strong basin-support evidence on `competitive_lv` with:
    - positive cosine separation
    - high support uniqueness (`mode_uniqueness_rate` near `1.0`)
    - clearly above-chance clustering / linear separability
  - The result is strong enough to support the paper’s “intrinsic-HD positive control” narrative without any new training.
- Failure criteria:
  - The offline pipeline cannot produce `competitive_lv` labels or metrics reliably.
  - All three roots show weak or contradictory support-alignment evidence, undermining the positive-control narrative.
  - The only workable path requires retraining rather than evaluation-only reuse.
- Queueing decision rule:
  - Do not queue anything unless the offline evaluation path fails locally or is too slow / brittle to finish from the existing checkpoints.

## Local QA Before Queueing

- Confirm the three root labels and their checkpoint paths match collected `competitive_lv` rows in the existing forecasting summaries.
- Smoke-test any new wrapper / patch with `--help` and a single-seed, `12`-trajectory run on CPU.
- Verify the smoke run writes the expected JSON/Markdown artifacts under the planned output root.
- Confirm the full task count before any batch execution:
  - `3` roots x `3` seeds = `9` checkpoint evaluations.
- Verify the block-diagonal comparator choice (`sc=0.003`) from the collected `competitive_lv` system rows before running the full pass.
- Keep the workflow offline-only if the full `9`-checkpoint evaluation completes locally; queue nothing in that case.
- If queueing becomes necessary after QA failure, record the exact reason the offline path failed and use `sbatch` only for the minimal evaluation-only job.

Performed:
- Verified checkpoint roots and timestamps against collected `competitive_lv` rows in:
  - `results/paper_followup_recipes_200k_20260309/collect/forecasting_summary.md`
  - `results/dense_lista_paper_rerun_stage4_20260309/collect/forecasting_summary.md`
- Verified the selected `checkpoint.pt` files exist under:
  - `/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/paper_followup_recipes/generic_sparse_ns200k_best/competitive_lv/dt_0p01/seed_{0,1,2}/.../checkpoint.pt`
  - `/network/scratch/l/lia/skae/dense_lista_paper_rerun_stage4_20260309/paper_rerun/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3/competitive_lv/dt_0p01/seed_{0,1,2}/.../checkpoint.pt`
  - `/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/paper_followup_recipes/lista_blockdiag_ns200k_denseopt_sc3em3/competitive_lv/dt_0p01/seed_{0,1,2}/.../checkpoint.pt`
- Added minimal evaluation-only helper:
  - `tools/paper_parallel_20260309_a_competitive_lv_support_alignment.py`
- Added minimal SLURM launcher:
  - `scripts/paper_parallel_20260309_a_competitive_lv_support_alignment.sh`
- Local checks run:
  - `uv run python -m py_compile /home/mila/l/lia/skae/tools/paper_parallel_20260309_a_competitive_lv_support_alignment.py`
  - `bash -n /home/mila/l/lia/skae/scripts/paper_parallel_20260309_a_competitive_lv_support_alignment.sh`
  - `rg -o -- '--entry' /home/mila/l/lia/skae/scripts/paper_parallel_20260309_a_competitive_lv_support_alignment.sh | wc -l` -> `9`
  - `sbatch --test-only /home/mila/l/lia/skae/scripts/paper_parallel_20260309_a_competitive_lv_support_alignment.sh`
- `sbatch --test-only` result:
  - Job `8914757`
  - partition `long-cpu`
  - single-node, `4` CPU task
- Deviation from the original offline-only preference:
  - Repeated login-node attempts to start checkpoint inference via `uv run python` stalled on `torch` import long before a smoke evaluation could complete, so I switched to the smallest evaluation-only compute-node fallback.

## What Was Queued

- Queued one evaluation-only SLURM job:
  - script: `scripts/paper_parallel_20260309_a_competitive_lv_support_alignment.sh`
  - command: `sbatch /home/mila/l/lia/skae/scripts/paper_parallel_20260309_a_competitive_lv_support_alignment.sh`
  - job ID: `8914758`
  - queue state at submission check:
    - `squeue -j 8914758 -o '%.18i %.9P %.40j %.8T %.10M %.6D %R'`
    - state: `PENDING`
    - partition: `long-cpu`
- Queued workload:
  - `9` checkpoint evaluations total
  - roots:
    - `generic_sparse_ns200k_best`
    - `lista_dense_promoted_stage4`
    - `lista_blockdiag_ns200k_denseopt_sc3em3`
  - seeds:
    - `0, 1, 2`
- Output root:
  - `results/paper_parallel_20260309_a_competitive_lv_support_alignment/`
- Expected artifacts on completion:
  - `results/paper_parallel_20260309_a_competitive_lv_support_alignment/per_run/<root>/seed_<n>/analysis.json`
  - `results/paper_parallel_20260309_a_competitive_lv_support_alignment/competitive_lv_support_alignment_summary.json`
  - `results/paper_parallel_20260309_a_competitive_lv_support_alignment/competitive_lv_support_alignment_summary.md`

## Results / Observations

- Initial job `8914758` FAILED with `AttributeError: 'BasinLabeledDataset' object has no attribute 'basin_counts'`.
  - Root cause: `BasinLabeledDataset` stores labels per-trajectory in `self.trajectories[i].final_basin`, not in a `basin_counts` dict.
  - Fix: replaced `dataset.basin_counts.items()` with a comprehension over `dataset.trajectories` in `tools/paper_parallel_20260309_a_competitive_lv_support_alignment.py` line 403.
  - Resubmitted as job `8914863` on March 9.
- Job `8914863` COMPLETED March 9. Results in `results/paper_parallel_20260309_a_competitive_lv_support_alignment/`.
- **INVALIDATED (March 10):** All results from this evaluation are INVALID. The evaluation ran on the old 1-basin `competitive_lv` (`INTERACTION_SCALE=0.35`). The `competitive_lv` config was subsequently updated to `INTERACTION_SCALE=0.70` producing 4 major basins. The degenerate single-basin metrics (ARI=1.0, purity=1.0) were artifacts of there being only 1 basin, not meaningful alignment signals.
- **Next steps:** After the competitive_lv multi-basin retrain (job `8922033`, 28 tasks, PENDING) completes, this entire evaluation must be re-run on the new 4-basin checkpoints. Do NOT re-use any results from this evaluation.

## Proposed Updates To Global Status Docs

- All prior competitive_lv support alignment results are invalid (1-basin data).
- After competitive_lv retrain (job `8922033`) completes, re-run this evaluation and update docs with the new 4-basin results.
- The competitive_lv support alignment question is now BLOCKED on retraining, not on evaluation tooling.
