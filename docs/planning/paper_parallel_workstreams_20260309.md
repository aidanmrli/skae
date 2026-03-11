# Paper Parallel Workstreams

Date: March 9, 2026

This file is the active coordination source of truth for the next paper-strengthening experiment phase. It supersedes ad hoc next-step bullets in the status docs for this specific workstream batch.

Primary goals:
- strengthen the paper's main positive claims with targeted follow-up experiments
- keep forecasting-first evaluation as the primary objective
- add basin-support evidence on intrinsic high-dimensional systems where possible
- only queue experiments after each workstream has a concrete plan and a local QA pass

Global rules for every workstream:
1. Read the current evidence first. Reuse completed runs whenever offline analysis is sufficient.
2. Write a concrete plan in the assigned status file before queueing:
   - objective and claim being tested
   - primary baselines and fairness controls
   - exact systems, seeds, horizons, and metrics
   - acceptance criteria and failure criteria
   - output root and artifact names
3. Check your work before queueing:
   - verify any task builder or collector locally
   - confirm task counts, root labels, and output directories
   - avoid the prior `sbatch --export` CSV bug pattern
   - use `uv run` for Python entry points
4. Queue only after the plan and QA notes are written down.
5. After queueing, update the assigned status file with:
   - what was run
   - exact scripts or commands used
   - task tables and output directories
   - SLURM job IDs
   - local QA checks performed
   - any deviations from plan
6. If a workstream produces decision-grade results, add a short proposed update for `docs/EXPERIMENTS.md` and `docs/PAPER_TRACK_STATUS.md` in the assigned status file.

Shared paper constraints:
- Do not rely on ground-truth basin labels or known basin counts for training-time method design.
- It is acceptable to use basin labels and counts for benchmark evaluation and diagnostics.
- Prioritize basin-support alignment over basin-block alignment.
- Prefer the cheapest experiment that can decisively support or refute the claim.

Immediate fairness blocker (March 11):
- Before freezing any paper claim involving `lista_blockdiag`, add a matched `generic_sparse + block_diagonal K` control everywhere `lista_blockdiag` currently appears.
- Tracking plan: [generic_sparse_blockdiag_fairness_plan_20260311.md](/home/mila/l/lia/skae/docs/planning/generic_sparse_blockdiag_fairness_plan_20260311.md)
- Required comparison set for any affected paper-facing study:
  - dense `generic_sparse` anchor
  - matched `generic_sparse_blockdiag*` structural control
  - matched `lista_blockdiag*` candidate
  - dense LISTA only when the existing paper narrative already needs the three-way comparison

## Subagent A: `competitive_lv` Support Alignment

Objective:
- Upgrade the basin-support story from Duffing/Lyapunov-only evidence to an intrinsic high-dimensional positive-control result on `competitive_lv`.

Primary inputs:
- `docs/PAPER_TRACK_STATUS.md`
- `docs/EXPERIMENTS.md`
- `docs/high_dim_supervisor_brief_20260305.md`
- `results/paper_followup_recipes_200k_20260309/`
- `results/dense_lista_paper_rerun_stage4_20260309/`
- `skae/data.py`
- `skae/basin_utils.py`
- `tools/evaluate_support_uniqueness.py`
- `tools/evaluate_basin_structure.py`
- `tools/evaluate_latent_basin_clustering.py`

Required tasks:
1. Audit whether existing tools already support `competitive_lv` basin labels; patch minimally if needed.
2. Define a fair three-way comparison on the existing headline roots:
   - `generic_sparse_ns200k_best`
   - `lista_dense_promoted_stage4`
   - best available block-diagonal comparator for `competitive_lv`
3. Report support uniqueness, cosine separation, and at least one clustering/separability metric.
4. Prefer offline evaluation on existing checkpoints before launching any retraining.
5. If a new eval sweep is needed, queue only the minimal targeted set.

Status log:
- [subagent_a_competitive_lv_support_alignment.md](/home/mila/l/lia/skae/docs/planning/paper_parallel_workstreams_20260309/subagent_a_competitive_lv_support_alignment.md)

## Subagent B: Kuramoto Support Alignment on Existing `200k` Runs

Objective:
- Test whether the current strongest intrinsic-HD forecasting result on Kuramoto also shows basin-discriminative supports.

Primary inputs:
- `docs/PAPER_TRACK_STATUS.md`
- `docs/EXPERIMENTS.md`
- `results/kuramoto_dt00625_200k_compare_20260308/`
- `results/kuramoto_n32_dt00625_200k_confirm_20260309/`
- `results/kuramoto_dimension_sweep_dt00625_200k_20260309/`
- `skae/data.py`
- `skae/basin_utils.py`
- `tools/evaluate_support_uniqueness.py`
- `tools/evaluate_latent_basin_clustering.py`

Required tasks:
1. Confirm how Kuramoto winding-number labels are computed and exposed.
2. Evaluate support uniqueness / cosine separation / clustering on the existing `dt=0.00625`, `200k` checkpoints.
3. Start with `N=16`; add `N=32` only if the `N=16` pipeline is clean.
4. Do not retrain unless the analysis pipeline itself requires missing artifacts.
5. If new evaluation jobs are queued, keep them narrow and evaluation-only.

Status log:
- [subagent_b_kuramoto_support_alignment.md](/home/mila/l/lia/skae/docs/planning/paper_parallel_workstreams_20260309/subagent_b_kuramoto_support_alignment.md)

## Subagent C: Fixed-Cadence Re-encoding Ablation

Objective:
- Replace or contextualize the mild oracle in `best-periodic` with a fixed-cadence evaluation that is closer to deployment.

Primary inputs:
- `docs/PAPER_TRACK_STATUS.md`
- `docs/EXPERIMENTS.md`
- `results/paper_followup_recipes_200k_20260309/`
- `results/kuramoto_dt00625_200k_compare_20260308/`
- `skae/evaluation.py`
- `tools/evaluate_checkpoints.py`
- `tools/collect_forecasting_roots.py`

Required tasks:
1. Define a fair fixed-cadence protocol:
   - either a single global cadence or a validation-chosen cadence without test leakage
2. Re-evaluate the headline low-dimensional dense-LISTA result and the headline Kuramoto result under that protocol.
3. Quantify how much of the current positive story survives without oracle best-period selection.
4. Prefer re-collection/re-scoring on existing checkpoints if possible.
5. If evaluation code changes are needed, validate them locally before queueing.

Status log:
- [subagent_c_fixed_cadence_reencoding.md](/home/mila/l/lia/skae/docs/planning/paper_parallel_workstreams_20260309/subagent_c_fixed_cadence_reencoding.md)

## Subagent D: More Seeds on Headline Positives

Objective:
- Tighten uncertainty on the exact results that would appear in the main text.

Primary inputs:
- `docs/PAPER_TRACK_STATUS.md`
- `docs/EXPERIMENTS.md`
- `results/dense_lista_paper_rerun_stage4_20260309/`
- `results/kuramoto_dt00625_200k_compare_20260308/`
- `results/kuramoto_n32_dt00625_200k_confirm_20260309/`
- existing queue scripts for dense reruns and Kuramoto targeted comparisons

Required tasks:
1. Choose the smallest high-value seed extensions:
   - dense LISTA low-dimensional headline set
   - Kuramoto `N=16`
   - Kuramoto `N=32`
2. Keep training recipes identical to the promoted settings.
3. Define exact additional seeds and the new target confidence summary.
4. Sanity-check task tables and root names before queueing.
5. Update the status file with the precise output roots and job IDs.

Status log:
- [subagent_d_more_seeds_headline_positives.md](/home/mila/l/lia/skae/docs/planning/paper_parallel_workstreams_20260309/subagent_d_more_seeds_headline_positives.md)

## Subagent E: Kuramoto Robustness Beyond the Default Setting

Objective:
- Test whether the Kuramoto block-diagonal win survives a mild change in regime rather than only the exact default setting.

Primary inputs:
- `docs/PAPER_TRACK_STATUS.md`
- `docs/EXPERIMENTS.md`
- `docs/planning/high_dim_benchmarks_plan.md`
- `results/kuramoto_dt00625_200k_compare_20260308/`
- `results/kuramoto_dimension_sweep_dt00625_200k_20260309/`
- `skae/data.py`
- existing Kuramoto queue scripts

Required tasks:
1. Pick the cheapest honest robustness lever:
   - mild frequency heterogeneity is preferred
   - only one additional coupling/topology variant if clearly justified
2. Keep the rest of the recipe fixed to the current Kuramoto winner comparison.
3. Include the MLP anchor and dense LISTA only if needed for the paper-facing comparison.
4. Define explicit promotion and failure criteria before queueing.
5. Smoke-check the new environment/task wiring locally.

Status log:
- [subagent_e_kuramoto_robustness_beyond_default.md](/home/mila/l/lia/skae/docs/planning/paper_parallel_workstreams_20260309/subagent_e_kuramoto_robustness_beyond_default.md)

## Subagent F: Support-Transition Dynamics

Objective:
- Produce appendix-grade evidence that supports are stable within regimes and switch near basin transitions.

Primary inputs:
- `docs/EXPERIMENTS.md`
- `results/dense_lista_paper_rerun_stage4_20260309/`
- `results/kuramoto_dt00625_200k_compare_20260308/`
- `results/*multiwell*`
- `skae/support_monitor.py`
- `skae/evaluation.py`

Required tasks:
1. Pick one clean low-dimensional transition system and one high-dimensional or moderate-dimensional system if feasible.
2. Define a support-stability metric along trajectories and a visualization plan.
3. Prefer offline analysis on existing checkpoints.
4. Queue only if the current tooling cannot generate the needed artifact.
5. Produce appendix-ready figures or summary tables, not just raw metrics.

Status log:
- [subagent_f_support_transition_dynamics.md](/home/mila/l/lia/skae/docs/planning/paper_parallel_workstreams_20260309/subagent_f_support_transition_dynamics.md)

## Subagent G: Checkpoint-Selection Ablation on Kuramoto

Objective:
- Quantify how much hard-system conclusions depend on validation-selected versus last-checkpoint model selection.

Primary inputs:
- `docs/PAPER_TRACK_STATUS.md`
- `docs/EXPERIMENTS.md`
- `results/intrinsic_hd_dt_rescue_20260308_rerun1/`
- `results/kuramoto_dt00625_200k_compare_20260308/`
- `evaluation_results_best.json`
- `evaluation_results_last.json`
- `tools/collect_forecasting_roots.py`

Required tasks:
1. Compare `best` versus `last` checkpoint selection on the current Kuramoto-focused roots.
2. Keep the analysis forecasting-first and horizon-specific.
3. Determine whether a revised model-selection rule is needed for the paper.
4. Prefer offline recollection over any retraining.
5. If code changes are needed for recollection, validate locally first.

Status log:
- [subagent_g_kuramoto_checkpoint_selection.md](/home/mila/l/lia/skae/docs/planning/paper_parallel_workstreams_20260309/subagent_g_kuramoto_checkpoint_selection.md)

## Subagent H: Label-Free Clustering Quality

Objective:
- Test whether latent supports or cosine features support label-free regime discovery on benchmark systems.

Primary inputs:
- `docs/PAPER_TRACK_STATUS.md`
- `docs/EXPERIMENTS.md`
- `results/paper_followup_recipes_200k_20260309/`
- `results/kuramoto_dt00625_200k_compare_20260308/`
- `tools/evaluate_latent_basin_clustering.py`
- `tools/evaluate_basin_structure.py`

Required tasks:
1. Choose a compact benchmark set:
   - Duffing
   - Lyapunov if compatible with current tooling
   - `competitive_lv`
   - Kuramoto
2. Compare at least one label-free clustering pipeline across the main encoder families.
3. Report ARI/NMI or equivalent against basin labels only for offline evaluation.
4. Prefer existing checkpoints and evaluation-only jobs.
5. Make clear whether this is appendix-only or ready for the main text.

Status log:
- [subagent_h_label_free_clustering_quality.md](/home/mila/l/lia/skae/docs/planning/paper_parallel_workstreams_20260309/subagent_h_label_free_clustering_quality.md)
