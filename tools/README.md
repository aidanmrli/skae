# Paper tooling

`tools/` contains the Python entry points that still reproduce, evaluate, or
render evidence used by `docs/neurips_sparse_koopman_multibasin.tex`. Run every
Python entry point through `uv run` on a compute-node allocation. The manuscript
and its appendix fragments remain the source of truth for the protocol and
claims; this file is only a code-navigation map.

## Training and shared evaluation

- `train.py`: shared training CLI used by the controlled and Dysts runners.
- `evaluate_checkpoints.py`: standardized checkpoint evaluation.
- `plot_training_metrics.py`: optional post-training plot imported by
  `train.py`.

## Frozen protocols and task construction

- `build_transition_rich_basin_partition_tasks.py`: exact 15-system, six-row
  controlled paper task table, with subset/seed/timestep repair knobs.
- `build_dysts_dt30_basinblock_tasks.py`: exact 10-system Dysts training table;
  hand-set lobe/scroll/equilibrium counts are explicitly diagnostic structure
  counts used only to size blocks.
- `build_dysts_long_horizon_eval_tasks.py`: H100--H5000 Dysts evaluation table;
  its root-spec input is explicit rather than a stale hard-coded campaign.
- `prebuild_dysts_cache.py`: deterministic Dysts cache materialization with
  safe paper defaults (`full`, (dt\times30)); systems remain explicit.
- `build_paper_baseline_tasks.py`: classical and local-linear baseline table.

Canonical settings shared by these builders live in
`skae/benchmarks/paper_protocol.py`; timestep resolution lives in
`skae/benchmarks/timesteps.py`.

## Evidence collection and tables

- `collect_forecasting_roots.py`, `compare_forecasting_roots.py`: controlled
  forecasting rows and paired comparisons.
- `collect_dysts_long_horizon_forecasting.py`: raw Dysts forecasting collector.
- `build_dysts_paper_evidence.py`: focused Dysts aggregation, system-level
  statistics, and appendix display from the frozen row packet.
- `reduce_transition_rich_interpretability_metrics.py` and
  `merge_transition_rich_interpretability_shards.py`: current support-alignment
  reduction pipeline.
- `build_table1_forecasting_support.py`, `paper_table_rendering.py`, and
  `freeze_main_paper_evidence.py`: freeze and render the main paper tables with
  provenance.
- `build_controlled_per_system_tables.py`: controlled per-system tables from
  the frozen evidence packet.
- `build_local_map_forecasting_tables.py`: staged local-map table rendering.

## Baselines and interventions

- `evaluate_classical_koopman_baselines.py` and
  `evaluate_mixture_local_linear_baselines.py`: paper baseline suite.
- `freeze_paper_baseline_evidence.py`: sanitize and hash-pin the external
  standalone-control row files into the compact paper evidence packet.
- `summarize_paper_baseline_suite.py`: regenerate or `--check` the baseline
  aggregate CSV sidecars and metadata from those frozen rows.
  The frozen schema distinguishes classical ordinary through-horizon means
  from mixture-local finite-prefix means; routine checks need no scratch access.
- `evaluate_support_coordinate_interventions.py` and
  `plot_support_coordinate_trajectories.py`: intervention evaluation and the
  checkpoint-dependent trajectory display.
- `freeze_intervention_evidence.py`: freeze intervention inputs and provenance.
- `build_intervention_artifacts.py`: verify and regenerate intervention curves
  and the H21 table from the frozen rows.
- `plot_multibasin_ground_truth_vector_fields.py`: controlled-system ground
  truth vector fields.

## Staged local-operator training

- `train_staged_support_family_local_k.py`: fixed two-stage paper trainer for
  F_abs-routed affine local maps and the only staged-training CLI.
- `staged_fabs_protocol.py`, `staged_fabs_tasks.py`, `staged_fabs_model.py`,
  `staged_fabs_training.py`, and `staged_fabs_io.py`: respectively freeze the
  scientific contract, parse task rows, define wrapped local maps, implement
  stage loops, and write schema-3 resumable artifacts. New checkpoints persist
  model/local optimizer state, training-generator state, and torch CPU/CUDA RNG
  state; legacy checkpoints without RNG state remain loadable but replay their
  seeded data streams. The source route fit deliberately duplicates one
  256-trajectory batch into 512 configured rows. Runtime support routing occurs
  before every latent transition; periodic decode--encode events refresh the
  latent but do not set the routing cadence. The selector uses the fixed
  32-start/H100--H1000 periodic contract.
- `reevaluate_staged_vs_global_wide_periodic.py`: staged/global reevaluation.
- `train_support_family_local_maps.py`: focused frozen-route construction and
  runtime assignment helper, including all-193-state clustering and modal-source
  representatives. It is not a standalone training entry point.
- `build_local_map_forecasting_tables.py`: hash-verifies the frozen 225-row
  packet and records the duplicated route fit, asymmetric checkpoint selectors,
  and 32/100 staged selector/evaluation overlap before rebuilding the TeX table.
