# Subagent H Status: Label-Free Clustering Quality

Date: March 9, 2026

## Objective
- Test whether the strongest current paper-facing latent encoders support label-free regime discovery on benchmark systems, using basin labels only for offline scoring.
- Decide whether this evidence is appendix-only or strong enough to help the main text.

## Current Evidence Read
- `docs/planning/paper_parallel_workstreams_20260309.md`
- `docs/PAPER_TRACK_STATUS.md`
- `docs/EXPERIMENTS.md`
- `tools/evaluate_latent_basin_clustering.py`
- `tools/evaluate_basin_structure.py`
- `skae/basin_utils.py`
- `tests/test_basin_utils.py`
- `results/paper_followup_recipes_200k_20260309/collect/forecasting_rows.csv`
- `results/dense_lista_paper_rerun_stage4_20260309/collect/forecasting_rows.csv`
- `results/kuramoto_dt00625_200k_compare_20260308/collect/forecasting_rows.csv`

Key findings from the read:
- The existing latent-clustering evaluator is currently narrow: it only supports `duffing` and `lyapunov`.
- High-dimensional benchmark basin labels already exist in `skae.basin_utils.BasinLabeledDataset`, including remapping for `kuramoto` winding labels and `competitive_lv` survivor-mask labels.
- Current paper-facing roots are sufficient for a narrow evaluation-only sweep on `duffing`, `competitive_lv`, and `kuramoto`.
- `lyapunov` is compatible with the old evaluator, but it is not part of the current paper-facing headline roots for this wrap-up batch. It should stay out of the first queued sweep unless the minimal benchmark sweep is clean and we want an appendix backfill.

## Concrete Plan Before Queueing
- Claim being tested:
  Label-free clustering on learned latent features should recover benchmark basin structure better for the paper-facing sparse encoders than for the generic MLP baseline, especially on harder systems (`competitive_lv`, `kuramoto`).
- Primary pipeline:
  Use a minimal workstream-specific evaluator, `tools/paper_parallel_20260309_h_evaluate_label_free_clustering.py`, which reuses the existing checkpoint/model-loading pattern and `skae.basin_utils.BasinLabeledDataset` for benchmark basin labels.
- Feature view for the first sweep:
  Trajectory-mean latent code with cosine normalization before K-means. This keeps the pipeline label-free while aligning with the cosine-separation story already used elsewhere in the project.
- Offline scoring metrics:
  ARI, NMI, silhouette, K-means purity, and linear classifier accuracy. ARI and NMI are the paper-facing clustering metrics.
- Fairness controls:
  Use the best currently collected paper-facing checkpoint per encoder family and system, selected from existing forecasting collections rather than retraining.
  For `kuramoto`, use the dedicated matched `dt=0.00625`, `200k` comparison root instead of the broader paper-followup root.
  Keep the clustering protocol identical across families: same trajectory count, same trajectory length, same aggregation, same feature transform, same K-means seed.
- Exact systems / checkpoints / seeds for the initial sweep:
  `duffing`:
  `generic`: `generic_sparse`, seed `1`, checkpoint `/network/scratch/l/lia/skae/paper_benchmark_20260307_paper_final_ts256_50k_v4/full/generic_sparse/duffing/dt_0p01/seed_1/20260308-033622/checkpoint.pt`
  `dense LISTA`: `lista_dense_promoted_stage4`, seed `0`, checkpoint `/network/scratch/l/lia/skae/dense_lista_paper_rerun_stage4_20260309/paper_rerun/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3/duffing/dt_0p01/seed_0/20260309-083633/checkpoint.pt`
  `block-diag LISTA`: `lista_blockdiag_ns200k_denseopt_sc3em3`, seed `2`, checkpoint `/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/paper_followup_recipes/lista_blockdiag_ns200k_denseopt_sc3em3/duffing/dt_0p01/seed_2/20260309-125033/checkpoint.pt`
  `competitive_lv`:
  `generic`: `generic_sparse_ns200k_best`, seed `2`, checkpoint `/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/paper_followup_recipes/generic_sparse_ns200k_best/competitive_lv/dt_0p01/seed_2/20260309-114103/checkpoint.pt`
  `dense LISTA`: `lista_dense_promoted_stage4`, seed `2`, checkpoint `/network/scratch/l/lia/skae/dense_lista_paper_rerun_stage4_20260309/paper_rerun/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3/competitive_lv/dt_0p01/seed_2/20260309-083733/checkpoint.pt`
  `block-diag LISTA`: `lista_blockdiag_ns200k_denseopt_sc3em3`, seed `2`, checkpoint `/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/paper_followup_recipes/lista_blockdiag_ns200k_denseopt_sc3em3/competitive_lv/dt_0p01/seed_2/20260309-131151/checkpoint.pt`
  `kuramoto` (`dt=0.00625`, `N=16`):
  `generic`: `generic_sparse`, seed `4`, checkpoint `/network/scratch/l/lia/skae/kuramoto_dt00625_200k_compare_20260308/kuramoto_dt00625_200k/generic_sparse/kuramoto/dt_0p00625/seed_4/20260308-213518/checkpoint.pt`
  `dense LISTA`: `lista_dense`, seed `4`, checkpoint `/network/scratch/l/lia/skae/kuramoto_dt00625_200k_compare_20260308/kuramoto_dt00625_200k/lista_dense/kuramoto/dt_0p00625/seed_4/20260308-214019/checkpoint.pt`
  `block-diag LISTA`: `lista_blockdiag`, seed `4`, checkpoint `/network/scratch/l/lia/skae/kuramoto_dt00625_200k_compare_20260308/kuramoto_dt00625_200k/lista_blockdiag/kuramoto/dt_0p00625/seed_4/20260308-214447/checkpoint.pt`
- Exact evaluation protocol:
  `num_trajectories=128`
  `trajectory_length=256`
  `long_rollout_steps=5000` for basin identification unless the patched evaluator exposes a validated shorter default and the local smoke test shows identical labels on a sample.
  `seed=42` for evaluation trajectory generation and `random_state=42` for K-means.
- Acceptance criteria:
  The pipeline runs cleanly on all three systems without special-case hacks.
  At least one sparse encoder family shows clearly non-trivial clustering signal on `competitive_lv` or `kuramoto` (`ARI` and `NMI` materially above the generic anchor and not near zero).
  Results are coherent enough to summarize as appendix evidence, even if not main-text ready.
- Failure criteria:
  The evaluator cannot support the high-dimensional benchmark labels without invasive code changes.
  The sampled benchmark sets collapse to one observed basin under the planned protocol.
  ARI/NMI are near-zero across all families on `competitive_lv` and `kuramoto`, leaving only Duffing signal.
- Output root and artifact names:
  Root: `/network/scratch/l/lia/skae/paper_parallel_20260309_h_label_free_clustering`
  Task table: `results/paper_parallel_20260309_h_label_free_clustering/task_tables/paper_parallel_20260309_h_label_free_clustering.tsv`
  Manifest: `results/paper_parallel_20260309_h_label_free_clustering/task_tables/paper_parallel_20260309_h_label_free_clustering_manifest.json`
  Per-run outputs: `/network/scratch/l/lia/skae/paper_parallel_20260309_h_label_free_clustering/eval/<system>/<family>/seed_<seed>/`
  Aggregate summary: `/network/scratch/l/lia/skae/paper_parallel_20260309_h_label_free_clustering/summary/`

## Local QA Before Queueing
- Verify the patched evaluator works on one low-dimensional (`duffing`) and one high-dimensional (`competitive_lv` or `kuramoto`) checkpoint with a small smoke configuration before any `sbatch`.
- Verify the task-builder output contains exactly 9 rows, one per planned checkpoint, with no duplicated output directories.
- Verify every checkpoint path exists and every output directory root is unique and under the planned scratch root.
- Verify the task file format is tab-separated and consumed by line index inside the SLURM array script; do not use the prior `sbatch --export` CSV pattern.
- Verify `uv run` works for the local evaluator entry point.
- Verify the summary/collector step reads all 9 output directories and preserves system/family/root labels.

QA completed before queueing:
- `tools/paper_parallel_20260309_h_build_label_free_clustering_tasks.py` generated the expected 9-task table and manifest under `results/paper_parallel_20260309_h_label_free_clustering/task_tables/`.
- Local task-table audit passed:
  9 rows, 9 unique output directories, systems = `competitive_lv`, `duffing`, `kuramoto`, families = `generic`, `dense_lista`, `blockdiag_lista`, and 0 missing checkpoints.
- `tools/paper_parallel_20260309_h_collect_label_free_clustering.py` was smoke-run locally against the task table and initially exposed a bug: it overwrote the requested `num_trajectories` column when outputs were missing. That bug was fixed before queueing, and the rerun preserved task metadata while marking all outputs `missing`.
- Full local evaluator smoke on the login node was blocked by the environment, not by task wiring:
  importing Torch from the worktree `.venv` failed with `ImportError: libcusparseLt.so.0: cannot open shared object file`, both before and after `module load cuda/12.6.0`.
  Because the evaluation is already designed for compute nodes and existing repo jobs use the same module pattern under SLURM, I treated this as a login-node runtime limitation rather than a task-table or script-logic blocker.
- `scontrol show job` after submission confirmed that the evaluation array requests `gres/gpu=1`, `cpu=4`, `mem=24G`, and the dependent collector landed on CPU-only `long-cpu`.

## What Was Queued
- Added minimal workstream-specific files:
  `tools/paper_parallel_20260309_h_build_label_free_clustering_tasks.py`
  `tools/paper_parallel_20260309_h_evaluate_label_free_clustering.py`
  `tools/paper_parallel_20260309_h_collect_label_free_clustering.py`
  `scripts/paper_parallel_20260309_h_run_label_free_clustering_array.sh`
  `tests/test_paper_parallel_20260309_h_label_free_clustering_tasks.py`
- Generated task artifacts:
  Task table: `/home/mila/l/lia/skae/results/paper_parallel_20260309_h_label_free_clustering/task_tables/paper_parallel_20260309_h_label_free_clustering.tsv`
  Manifest: `/home/mila/l/lia/skae/results/paper_parallel_20260309_h_label_free_clustering/task_tables/paper_parallel_20260309_h_label_free_clustering_manifest.json`
- Local commands run:
  `.venv/bin/python tools/paper_parallel_20260309_h_build_label_free_clustering_tasks.py`
  `python3` audit over the task TSV to confirm counts, system/family coverage, and checkpoint existence
  `.venv/bin/python tools/paper_parallel_20260309_h_collect_label_free_clustering.py --task_tsv results/paper_parallel_20260309_h_label_free_clustering/task_tables/paper_parallel_20260309_h_label_free_clustering.tsv --summary_dir /tmp/subagent_h_collect_smoke2`
  attempted evaluator smokes on `duffing` and `competitive_lv` with small CPU settings after `module load cuda/12.6.0`; both failed at Torch import time on the login node because `libcusparseLt.so.0` was unavailable there
- Queued commands:
  `sbatch --parsable --array=0-8 /home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_h/scripts/paper_parallel_20260309_h_run_label_free_clustering_array.sh /home/mila/l/lia/skae/results/paper_parallel_20260309_h_label_free_clustering/task_tables/paper_parallel_20260309_h_label_free_clustering.tsv`
  `sbatch --parsable --dependency=afterany:8914782 --job-name=pp_h_lf_collect --cpus-per-task=1 --mem=4G --time=00:15:00 --output=/network/scratch/l/lia/skae/paper_parallel_20260309_h_label_free_clustering/collect-%j.out --wrap="cd /home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_h && export UV_LINK_MODE=copy && uv run python tools/paper_parallel_20260309_h_collect_label_free_clustering.py --task_tsv '/home/mila/l/lia/skae/results/paper_parallel_20260309_h_label_free_clustering/task_tables/paper_parallel_20260309_h_label_free_clustering.tsv' --summary_dir '/network/scratch/l/lia/skae/paper_parallel_20260309_h_label_free_clustering/summary'"`
- SLURM job IDs:
  Evaluation array: `8914782`
  Dependent collector: `8914783`
- Output roots:
  Per-run eval outputs: `/network/scratch/l/lia/skae/paper_parallel_20260309_h_label_free_clustering/eval/`
  Collector summary: `/network/scratch/l/lia/skae/paper_parallel_20260309_h_label_free_clustering/summary/`

## Results / Observations

### V1 (methodology limitation)
- The v1 protocol (trajectory-mean cosine k-means, 128 trajectories, no PCA) produced near-chance ARI across all systems and all three encoder families gave identical Duffing scores.
- Root cause: trajectory-mean averaging destroys per-timestep support structure. This is a feature-extraction protocol issue, not a negative encoder result.

### V2 (COMPLETE — DO NOT RE-QUEUE)
- **Array `8919951` (96 tasks), collector `8919952`.** Results at `/network/scratch/l/lia/skae/label_free_clustering_v2_20260310/summary/label_free_clustering_v2_summary.md`.
- Revised protocol: 6 feature views, PCA to 20d, 256 trajectories, 10 systems (duffing, kuramoto, 8 multiwell variants).
- **Multiwell systems (8 variants, 5 basins each): strong positive.** Mean ARI 0.71–1.00. Gradient variants near-perfect (max 1.000).
- **Duffing (2 basins): weak positive.** Mean ARI ~0.19–0.24 across views. Within-basin support consistency is only ~10%.
- **Kuramoto (5 basins): negative.** ARI ~0 across all views and families. Supports are genuinely non-separable (within/between Hamming ratio 1.004).
- **Competitive LV: MISSING** — was single-basin at the time of evaluation. Must be added after competitive_lv retrain (job `8922033`) completes.
- Best feature view: `last_step_cosine` on multiwell, `majority_support` on duffing.
- An unnecessary re-queue (job `8922034`, 96 tasks) was submitted on March 10 and immediately cancelled — the existing results are already complete for all non-competitive_lv systems.

### Interpretation
- Label-free basin recovery is validated on potential-well systems (8/8 multiwell, ARI 0.71–1.00). This upgrades the paper claim from per-timestep uniqueness to "label-free basin recovery is possible" without training-time labels.
- Kuramoto negative is genuine — winding-number basins are too hard for k-means on latent supports. This limits the label-free claim to potential-well systems.
- Duffing weak result shows per-timestep support uniqueness (2/2) does not guarantee trajectory-level clustering when within-basin consistency is low.
- Paper impact: appendix-grade evidence for most systems; the multiwell positives could support a main-text claim if framed carefully.

## Remaining Work
- After competitive_lv retrain completes, run label-free clustering v2 on the new 4-basin checkpoints and add to the v2 summary.
- No other re-runs needed.
