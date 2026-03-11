# Subagent B Status: Kuramoto Support Alignment

Date: March 9, 2026

## Objective

- Test whether the current rescued Kuramoto `dt=0.00625`, `200k` forecasting checkpoints also exhibit basin-discriminative latent supports, with `N=16` as the required first target and `N=32` only if the `N=16` analysis pipeline is clean.
- Claim under test: the same setting that yields decision-grade forecasting on intrinsic high-dimensional Kuramoto, especially `lista_blockdiag`, should also show measurable basin-support alignment under winding-number evaluation.

## Current Evidence Read

- [docs/planning/paper_parallel_workstreams_20260309.md](/home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_b/docs/planning/paper_parallel_workstreams_20260309.md)
- [docs/planning/paper_parallel_workstreams_20260309/subagent_b_kuramoto_support_alignment.md](/home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_b/docs/planning/paper_parallel_workstreams_20260309/subagent_b_kuramoto_support_alignment.md)
- [docs/PAPER_TRACK_STATUS.md](/home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_b/docs/PAPER_TRACK_STATUS.md) Kuramoto sections
- [docs/EXPERIMENTS.md](/home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_b/docs/EXPERIMENTS.md) Kuramoto sections
- [docs/high_dim_supervisor_brief_20260305.md](/home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_b/docs/high_dim_supervisor_brief_20260305.md) Kuramoto guidance
- [results/kuramoto_dt00625_200k_compare_20260308/collect/forecasting_summary.md](/home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_b/results/kuramoto_dt00625_200k_compare_20260308/collect/forecasting_summary.md)
- [results/kuramoto_n32_dt00625_200k_confirm_20260309/forecasting_summary.md](/home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_b/results/kuramoto_n32_dt00625_200k_confirm_20260309/forecasting_summary.md)
- [results/kuramoto_dimension_sweep_dt00625_200k_20260309/collect/kuramoto_dimension_summary.md](/home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_b/results/kuramoto_dimension_sweep_dt00625_200k_20260309/collect/kuramoto_dimension_summary.md)
- [skae/data.py](/home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_b/skae/data.py) `KuramotoOscillators.basin_label`
- [skae/basin_utils.py](/home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_b/skae/basin_utils.py)
- [tools/evaluate_support_uniqueness.py](/home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_b/tools/evaluate_support_uniqueness.py)
- [tools/evaluate_latent_basin_clustering.py](/home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_b/tools/evaluate_latent_basin_clustering.py)
- [tests/test_basin_utils.py](/home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_b/tests/test_basin_utils.py)

Current evidence summary:
- Kuramoto winding labels are already exposed through `KuramotoOscillators.basin_label`, which computes the signed winding number from wrapped phase differences after a long rollout.
- `BasinLabeledDataset` already supports Kuramoto by remapping arbitrary signed winding numbers to contiguous basin IDs and naming them `Winding q=...`.
- `tools/evaluate_support_uniqueness.py` already uses `BasinLabeledDataset`, so support uniqueness and cosine separation should work on Kuramoto without retraining.
- `tools/evaluate_latent_basin_clustering.py` still carries a Duffing/Lyapunov-only local dataset implementation, so clustering on Kuramoto likely requires a minimal evaluation-only patch if that tool is needed.
- The current forecasting-positive Kuramoto settings are:
  - `N=16`: `results/kuramoto_dt00625_200k_compare_20260308`, where `lista_blockdiag` is the strongest `H1000` median best-periodic model (`6.9803`) vs `generic_sparse` (`27.0193`) and dense LISTA (`13.8445`).
  - `N=32`: `results/kuramoto_dimension_sweep_dt00625_200k_20260309` and `results/kuramoto_n32_dt00625_200k_confirm_20260309`, where both `generic_sparse` and `lista_blockdiag` are good and `lista_blockdiag` remains slightly better at `H1000`.
- Existing Kuramoto run directories contain `checkpoint.pt`, `last.pt`, and `evaluation_results_best.json`, but no precomputed support-uniqueness or latent-clustering outputs were found under the inspected roots.

## Concrete Plan Before Queueing

- Objective and claim:
  - Measure whether basin-support alignment appears on the same rescued Kuramoto checkpoints that support the forecasting story, not on new retrains.
- Primary baselines and fairness controls:
  - Use the existing `dt=0.00625`, `200k` checkpoints only.
  - Compare the same three model families already used in the paper-facing Kuramoto summaries:
    - `generic_sparse`
    - `lista_blockdiag`
    - dense LISTA comparator from the same root (`lista_dense` for `N=16`, `lista_dense_promoted_n32` for `N=32` if reached)
  - Use the same checkpoint selection convention as the paper summaries: `checkpoint.pt` as the best-checkpoint artifact.
  - Use the same number of trajectories, trajectory length, and basin-label rollout across models within each `N`.
- Exact systems, seeds, horizons, and metrics:
  - System: `kuramoto`
  - Required pass: `N=16`, `dt=0.00625`, `200k`, seeds `0..4`
  - Optional pass if the `N=16` pipeline is clean: `N=32`, `dt=0.00625`, `200k`, seeds `0..4`
  - Forecasting context recorded from existing summaries: `H100`, `H500`, `H1000` best-periodic medians from the completed collections; no new forecasting job is planned.
  - Primary support-alignment metrics:
    - support uniqueness rate
    - mode collision pairs
    - mean basin consistency
    - mean pairwise Jaccard
    - mean intra-basin cosine
    - mean inter-basin cosine
    - cosine separation score
  - Clustering/separability metric target:
    - at least one of silhouette score, adjusted Rand index, k-means purity, or linear classifier accuracy from a Kuramoto-capable clustering path
- Acceptance criteria:
  - The `N=16` evaluation path runs locally on existing checkpoints with no retraining.
  - At least one Kuramoto-positive model, ideally `lista_blockdiag`, shows nontrivial basin-support structure by cosine separation and either unique mode supports or a clustering/separability metric that is clearly above chance.
  - Outputs are written to a dedicated paper-parallel result root with explicit `n16` and optional `n32` subdirectories.
- Failure criteria:
  - Kuramoto labels do not propagate cleanly into the current evaluation tools.
  - The clustering tool cannot be patched minimally for Kuramoto.
  - Existing checkpoints are missing or incompatible.
  - Support metrics are entirely degenerate across all seeds and models, yielding no paper-strengthening evidence.
- Output root and artifact names:
  - Local analysis root: `/home/mila/l/lia/parallel_agents/paper_parallel_20260309/subagent_b/results/paper_parallel_20260309_b_kuramoto_support_alignment`
  - If code changes are required, use uniquely prefixed files named `paper_parallel_20260309_b_*`
  - Planned artifact layout:
    - `.../n16/<root_label>/seed_<k>/support_uniqueness.json`
    - `.../n16/<root_label>/seed_<k>/cosine_metrics.json`
    - `.../n16/<root_label>/seed_<k>/cosine_diagnostics.json` if useful
    - `.../n16/<root_label>/seed_<k>/analysis_results.json` for clustering if enabled
    - matching `n32/` artifacts only if the `N=16` pass is clean
- Execution order:
  - First confirm label plumbing and tool compatibility.
  - Then run a local smoke test on one `N=16` checkpoint.
  - If the smoke test passes, run the full `N=16` offline evaluation locally if feasible.
  - Patch only the minimal clustering path if needed.
  - Queue only a narrow evaluation-only job if local execution is too slow or if a batch collector is needed after the local QA pass.

## Local QA Before Queueing

- [ ] Confirm `KuramotoOscillators.basin_label` and `BasinLabeledDataset` agree on signed winding-number labeling and remapping.
- [ ] Confirm `tests/test_basin_utils.py` contains Kuramoto coverage and run the focused test locally if needed.
- [ ] Confirm `tools/evaluate_support_uniqueness.py` loads `checkpoint.pt` from the targeted runs and works with `system=kuramoto`.
- [ ] Smoke-test one `N=16` checkpoint locally with `uv run` before any batch job.
- [ ] Verify output directories are unique and under `results/paper_parallel_20260309_b_kuramoto_support_alignment`.
- [ ] Verify checkpoint counts and exact run paths for the targeted `N=16` roots and seeds.
- [ ] Avoid `sbatch --export` CSV patterns entirely if any SLURM job is needed.
- [ ] If clustering needs a patch, validate the patched tool locally on one seed before any queue action.
- [ ] If an `N=32` pass is attempted, do it only after `N=16` outputs are complete and clean.
- [ ] Before queueing anything, record the exact command(s), task counts, and expected output directories here.

## What Was Queued

- Nothing. This workstream is closed without queueing.

## Results / Observations

- **Closed as redundant (March 10, 2026).**
- The label-free clustering v2 sweep (Subagent H v2, completed March 10) already ran on the exact same checkpoints this workstream targeted: `kuramoto_dt00625_200k_compare_20260308`, seeds 0–4, all 3 families (`generic_sparse`, `lista_dense`, `lista_blockdiag`).
- v2 results are conclusively negative for Kuramoto support alignment:
  - ARI ~0 across all views and all families
  - within-basin vs between-basin Hamming distance ratio = 1.004 (flat, non-separable)
  - every trajectory has a unique support pattern (~93 active dims, identical across all 5 basins)
  - purity = 0.5859 = majority class fraction (random w.r.t. basin structure)
- Running support uniqueness / cosine separation via `evaluate_support_uniqueness.py` on these same checkpoints would confirm the same negative, since the underlying representation is non-separable.
- The Kuramoto story in the paper is purely a **forecasting** result (block-diagonal LISTA rescues long-horizon rollouts), not a basin-support alignment result.

## Proposed Updates To Global Status Docs

- No update needed — the Kuramoto negative is already documented in `EXPERIMENTS.md` and `PAPER_TRACK_STATUS.md` from the v2 label-free clustering results.
