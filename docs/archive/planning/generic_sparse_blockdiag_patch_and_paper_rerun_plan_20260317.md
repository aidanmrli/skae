# `GenericKM` Structured-`K` Patch and Paper-Facing Rerun Plan

Date: March 17, 2026

## Summary

The March 17 audit showed that the historical `generic_sparse + block_diagonal K`
controls are invalid because `GenericKM` ignores `MODEL.K_STRUCTURE` and always
learns a dense latent transition.

This plan fixes that implementation gap, adds regression tests that enforce
Koopman-structure contracts at the model level, and reruns only the invalid
paper-facing MLP `+ block-K` artifacts first.

The immediate goal is not a full historical cleanup. The first pass is limited
to the current paper-facing invalid roots:

- targeted Kuramoto fairness control
- corrected 4-basin `competitive_lv` MLP `+ block-K` control
- higher-basin Hopfield quarter-`dt` MLP `+ block-K` control
- higher-basin fixed-system `competitive_lv` MLP `+ block-K` follow-up

Other historical `GenericKM + structured K` runs should be redone later, but
they are explicitly out of scope for this first repair wave.

## Objectives

1. Make `GenericKM` honor `MODEL.K_STRUCTURE` and `K_BLOCK_SIZE`.
2. Add regression tests that fail if any structured-capable Koopman model
   violates its intended transition structure.
3. Preserve existing dense-`K` behavior for `GenericKM`.
4. Refresh the invalid paper-facing MLP `+ block-K` rows with valid reruns.

## Model/Test Design

### Shared structure-invariant tests

Add a new shared test layer, preferably in
[tests/test_k_structure.py](/home/mila/l/lia/skae/tests/test_k_structure.py),
with helpers that operate on the public `kmatrix()` / `step_latent()` contract.

Required checks:

- `kmatrix()` has the expected structural mask
- `step_latent(y)` matches `y @ kmatrix()` numerically
- active Koopman parameters receive gradients

Covered model families:

- `GenericKM`
  - `dense`
  - `diagonal`
  - `block_diagonal`
- `LISTAKM`
  - `dense`
  - `diagonal`
  - `block_diagonal`
- `StructuredLISTAKM`
  - arrowhead structure assembled from `K_global`, `K_coupling`, and `K_basin`

This keeps one canonical structural test contract across all structured-capable
models instead of scattering model-specific assertions.

### Existing test updates

Update [tests/test_model.py](/home/mila/l/lia/skae/tests/test_model.py) only
where it currently hardcodes dense-only assumptions, especially:

- any gradient assertion that assumes `model.kmat` always exists
- any direct dense-only Koopman inspection

Dense-path behavior tests should remain unchanged.

## `GenericKM` implementation changes

Patch [skae/model.py](/home/mila/l/lia/skae/skae/model.py) so `GenericKM`
matches the same dense / diagonal / block-diagonal semantics already used by
`LISTAKM`.

Required behavior:

- set `self._k_structure = cfg.MODEL.K_STRUCTURE`
- for `dense`, keep `self.kmat`
- for `diagonal`, store `self.kmat_diag`
- for `block_diagonal`, store `self.kmat_blocks` and optional
  `self.kmat_remainder`
- use the same `K_BLOCK_SIZE` policy as `LISTAKM`
- implement `kmatrix()` and `step_latent()` consistently with that structure

Must not change:

- MLP encoder/decoder architecture
- latent normalization behavior
- loss definitions
- dense `GenericKM` checkpoint shape and behavior

## Compatibility policy

Dense `GenericKM` checkpoints remain backward compatible.

Old invalid historical `GenericKM` runs that recorded
`K_STRUCTURE=block_diagonal` but actually saved dense `kmat` are **not**
reinterpreted as valid structured checkpoints. They remain invalid provenance
artifacts and should fail clearly if loaded into the repaired structured-`K`
path.

One downstream utility also needs a small compatibility update:

- [tools/analyze_k_eigenvalues.py](/home/mila/l/lia/skae/tools/analyze_k_eigenvalues.py)

It currently assumes only `LISTAKM` can be structured and treats all other
models as dense. After the patch it should inspect `_k_structure` on repaired
`GenericKM` models too.

## Paper-Facing Rerun Scope

### 1. Targeted Kuramoto fairness control

Refresh the historical root under
`results/kuramoto_gs_blockdiag_fairness_20260310`.

Rerun:

- `generic_sparse_blockdiag`
- `env=kuramoto`
- `N=16`
- `dt=0.00625`
- `num_steps=200000`
- seeds `0-4`

Use the existing dedicated launcher shape in
[scripts/queue_kuramoto_gs_blockdiag_fairness.sh](/home/mila/l/lia/skae/scripts/queue_kuramoto_gs_blockdiag_fairness.sh),
but write to a fresh dated output root.

Refresh the collector that feeds:

- the Kuramoto hard-system table
- the block-`K` fairness summary table

### 2. Corrected 4-basin `competitive_lv`

Refresh the historical MLP `+ block-K` control used by:

- `results/clv_generic_sparse_blockdiag_20260310`
- `results/clv_15seed_extension_20260311`

Rerun:

- `generic_sparse_blockdiag_200k`
- `env=competitive_lv`
- `dt=0.01`
- `num_steps=200000`
- seeds `0-14`

Do this as one fresh valid 15-seed campaign rather than mixing new and old
sub-collectors.

Then:

- rebuild the paper-facing full 15-seed collector
- derive the original 3-seed fairness-control row from seeds `0-2` of the same
  valid rerun

### 3. Higher-basin Hopfield quarter-`dt`

Refresh the historical MLP `+ block-K` control used by:

- `results/hopfield_gs_blockdiag_fairness_dt0p0015625_20260311`
- the quarter-`dt` Hopfield paper table
- the fairness-summary `P=14` row

Rerun:

- `generic_sparse_blockdiag`
- `env=hopfield`
- `N=64`
- `P in {8,10,12,14,16}`
- `dt=0.0015625`
- `num_steps=200000`
- seeds `0-2`

Refresh:

- aggregate collector
- per-pattern collector

### 4. Higher-basin fixed-system `competitive_lv`

Refresh the historical MLP `+ block-K` controls cited in the smaller-`dt`
follow-up table under `results/clv_high_basin_20260310` and
`results/clv_high_basin_dt_halving_20260311`.

Rerun only the MLP structured-control rows for the paper-cited fixed systems:

- `species=12`, `interaction_scale=0.80`, `system_seed=0`
- `species=15`, `interaction_scale=0.83`, `system_seed=0`
- `dt in {0.005, 0.0025}`
- `num_steps=200000`
- seeds `0-2`

Do not rerun the dense or LISTA families in this pass.

## Explicit non-goals for this pass

Do not rerun:

- dense baselines
- LISTA roots
- targeted Hopfield `N=16`, `dt=0.00625` runs
- Kuramoto `N=32` follow-up
- Kuramoto uniform-frequency-spread follow-up
- Kuramoto dimension sweep
- full hard-system parity Stage 1/2 historical roots
- broader historical `GenericKM + structured K` artifacts outside the current
  paper-facing tables

Those broader invalid artifacts should be cleaned up later, but they are not on
the critical path for the current paper packet.

## Local QA before queueing

Minimum required test run:

```bash
uv run pytest tests/test_k_structure.py tests/test_model.py -q
```

If any task-builder or queue wrapper is touched, also run the smallest relevant
smoke test or builder-generation check for that script.

Queue QA requirements:

- task tables show `k_structure=block_diagonal`
- task tables show `k_block_size=16`
- rerun manifests include only the intended MLP structured-control roots
- collectors point only at fresh rerun roots, never at the invalid historical
  directories

## Acceptance criteria

Implementation is complete when all of the following hold:

1. `GenericKM` passes dense / diagonal / block-diagonal structure tests.
2. `LISTAKM` and `StructuredLISTAKM` pass the shared structure-invariant tests.
3. Dense `GenericKM` behavior remains unchanged.
4. The repaired paper-facing MLP `+ block-K` reruns are collected into fresh
   valid roots.
5. Paper-facing docs can replace the current invalid MLP `+ block-K` numbers
   with valid rerun results.

## Documentation follow-through after reruns

After valid reruns are collected:

- update [docs/EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md)
- update [docs/PAPER_TRACK_STATUS.md](/home/mila/l/lia/skae/docs/PAPER_TRACK_STATUS.md)
- update the affected paper tables in
  [docs/review_main_results_tables_20260314.tex](/home/mila/l/lia/skae/docs/review_main_results_tables_20260314.tex)

Until then, the historical MLP `+ block-K` rows remain invalid provenance only.

## Queue Plan and QA Notes

The implemented launcher layer is intentionally narrow on the training side and
mixed only where required on the collection side: each wrapper below creates a
fresh dated output root and writes a task TSV that includes only the intended
MLP structured-control reruns for this repair wave. The collectors then join
those fresh rerun roots with the already-valid baseline / LISTA roots that feed
the current paper tables, while explicitly excluding the invalid historical MLP
`+ block-K` directories.

Fresh wrappers:

- [scripts/queue_kuramoto_gs_blockdiag_fairness_20260317.sh](/home/mila/l/lia/skae/scripts/queue_kuramoto_gs_blockdiag_fairness_20260317.sh)
- [scripts/queue_clv_15seed_extension_20260317.sh](/home/mila/l/lia/skae/scripts/queue_clv_15seed_extension_20260317.sh)
- [scripts/queue_hopfield_gs_blockdiag_fairness_20260317.sh](/home/mila/l/lia/skae/scripts/queue_hopfield_gs_blockdiag_fairness_20260317.sh)
- [scripts/queue_clv_high_basin_blockdiag_20260317.sh](/home/mila/l/lia/skae/scripts/queue_clv_high_basin_blockdiag_20260317.sh)

Fresh dated output roots:

- `results/kuramoto_gs_blockdiag_fairness_20260317`
- `results/clv_15seed_extension_20260317`
- `results/hopfield_gs_blockdiag_fairness_dt0p0015625_20260317`
- `results/clv_high_basin_blockdiag_20260317`

Collector layout:

- Kuramoto: fresh `generic_sparse_blockdiag` root plus the existing valid
  `generic_sparse` and `lista_blockdiag` roots from
  `/network/scratch/l/lia/skae/kuramoto_dt00625_200k_compare_20260308`.
- Hopfield quarter-`dt`: fresh `generic_sparse_blockdiag` root plus the
  existing valid `generic_sparse`, `lista_dense_promoted_stage4`, and
  `lista_blockdiag_targeted` roots from
  `/network/scratch/l/lia/skae/hopfield_basin_sweep_n64_dt0p0015625_200k_20260311`.
- Corrected 4-basin `competitive_lv` 15-seed rebuild: fresh
  `generic_sparse_blockdiag_200k` root plus the existing valid
  `generic_sparse_ns200k_best`, `lista_dense_promoted_stage4`,
  `lista_blockdiag_ns200k_denseopt_sc3em3`, and
  `lista_blockdiag_ns200k_denseopt_sc6em3` roots from
  `/network/scratch/l/lia/skae/competitive_lv_multibas_retrain_20260310` and
  `/network/scratch/l/lia/skae/clv_15seed_extension_20260311`.
- Fixed-system high-basin `competitive_lv`: one fresh `generic_sparse_blockdiag_200k`
  rerun per paper-cited setting, each collected and compared against the
  existing valid `generic_sparse_200k`, `lista_dense_200k`, and
  `lista_blockdiag_200k` roots from
  `/network/scratch/l/lia/skae/clv_high_basin_dt_halving_20260311`.

Dry-run QA path before submission:

1. `bash -n` each wrapper script.
2. Run each wrapper with `DRY_RUN=1` so it writes the TSV and root-spec files but skips `sbatch`.
3. Verify the generated TSVs contain the intended structural settings:
   - `k_structure=block_diagonal`
   - `k_block_size=16`
4. Verify the root-spec files use the fresh output roots for the repaired MLP
   controls and never reference the invalid historical MLP `+ block-K`
   directories.
5. Hand the wrappers to the main queue step with `DRY_RUN=0` once the local smoke checks pass.

## Implementation, Submission, and First Live Verification

Implementation status:

- [skae/model.py](/home/mila/l/lia/skae/skae/model.py) now makes `GenericKM`
  honor `MODEL.K_STRUCTURE` for `dense`, `diagonal`, and `block_diagonal`.
- [tests/test_k_structure.py](/home/mila/l/lia/skae/tests/test_k_structure.py)
  adds the shared structure-invariant test contract across `GenericKM`,
  `LISTAKM`, and `StructuredLISTAKM`.
- [tools/analyze_k_eigenvalues.py](/home/mila/l/lia/skae/tools/analyze_k_eigenvalues.py)
  now treats repaired `GenericKM` checkpoints as structured when appropriate.

Local QA completed before submission:

- `uv run python -m pytest tests/test_k_structure.py tests/test_model.py -q`
  passed with `61 passed`.
- `bash -n` passed for all four fresh wrappers:
  - [scripts/queue_kuramoto_gs_blockdiag_fairness_20260317.sh](/home/mila/l/lia/skae/scripts/queue_kuramoto_gs_blockdiag_fairness_20260317.sh)
  - [scripts/queue_hopfield_gs_blockdiag_fairness_20260317.sh](/home/mila/l/lia/skae/scripts/queue_hopfield_gs_blockdiag_fairness_20260317.sh)
  - [scripts/queue_clv_15seed_extension_20260317.sh](/home/mila/l/lia/skae/scripts/queue_clv_15seed_extension_20260317.sh)
  - [scripts/queue_clv_high_basin_blockdiag_20260317.sh](/home/mila/l/lia/skae/scripts/queue_clv_high_basin_blockdiag_20260317.sh)
- `DRY_RUN=1` generation checks passed for all four wrappers.
- Generated task TSVs and root-spec files were checked locally to confirm:
  - every repaired training row keeps `k_structure=block_diagonal`
  - every repaired training row keeps `k_block_size=16`
  - collectors join only fresh repaired MLP roots with already-valid anchors
  - invalid historical MLP `+ block-K` directories are excluded

Submitted SLURM wave:

- Kuramoto fairness rerun:
  - array `8972079`
  - collect `8972081`
  - compare `8972085`
- Hopfield quarter-`dt` fairness rerun:
  - array `8972080`
  - collect `8972082`
  - compare `8972086`
- Corrected 4-basin `competitive_lv` 15-seed rebuild:
  - array `8972083`
  - collect `8972087`
  - compare `8972089`
- High-basin fixed-system `competitive_lv` rerun:
  - array `8972084`
  - `6bas_200k_dt0p005`: collect `8972088`, compare `8972090`
  - `6bas_200k_dt0p0025`: collect `8972091`, compare `8972092`
  - `8bas_200k_dt0p005`: collect `8972093`, compare `8972094`
  - `8bas_200k_dt0p0025`: collect `8972095`, compare `8972096`

First live verification (`2026-03-17 19:44 EDT`):

- `squeue` and `sacct` show Kuramoto array `8972079` has started cleanly:
  tasks `0-2` are running as subjobs `8972098/8972099/8972100`, while tasks
  `3-4` are still pending on priority.
- Hopfield quarter-`dt` (`8972080`), corrected-CLV 15-seed (`8972083`), and
  high-basin CLV (`8972084`) are still pending on priority. Their dependent
  collectors / comparisons remain pending on dependencies as expected.
- The Kuramoto SLURM stdout / stderr files show the runner banner plus only the
  expected `VIRTUAL_ENV` mismatch warning from `uv`; there is no traceback,
  import failure, or configuration error.
- Fresh Kuramoto run directories already exist under
  `/network/scratch/l/lia/skae/kuramoto_gs_blockdiag_fairness_20260317/...`
  and contain `config.json`, `checkpoint.pt`, `last.pt`, and
  `metrics_history.jsonl`.
- Live metric polling confirmed real training progress rather than startup-only
  execution. The last recorded steps advanced from `7045/8466/9373` to
  `8195/9787/10836` across the three running seeds during repeated checks.

Current status:

- The implementation and queue-launch acceptance criteria are met.
- The paper-facing fairness-control blocker is now in the “fresh collected
  summaries ready for interpretation” state.
- Historical MLP `+ block-K` rows remain invalid until the fresh collectors
  complete and replace them with valid rerun summaries.

Completion refresh (`2026-03-19 14:50 EDT`):

- No live SLURM jobs remain.
- Training arrays `8972079/8972080/8972083/8972084` and dependent collect /
  compare jobs `8972081-8972096` all completed with `ExitCode 0:0`.
- Fresh collected outputs now exist at:
  - [results/kuramoto_gs_blockdiag_fairness_20260317/collect/paper_benchmark_summary.md](/home/mila/l/lia/skae/results/kuramoto_gs_blockdiag_fairness_20260317/collect/paper_benchmark_summary.md)
  - [results/hopfield_gs_blockdiag_fairness_dt0p0015625_20260317/collect/paper_benchmark_summary.md](/home/mila/l/lia/skae/results/hopfield_gs_blockdiag_fairness_dt0p0015625_20260317/collect/paper_benchmark_summary.md)
  - [results/clv_15seed_extension_20260317/collect_200k_15seed_full/paper_benchmark_summary.md](/home/mila/l/lia/skae/results/clv_15seed_extension_20260317/collect_200k_15seed_full/paper_benchmark_summary.md)
  - [results/clv_high_basin_blockdiag_20260317/collect/6bas_200k_dt0p005/paper_benchmark_summary.md](/home/mila/l/lia/skae/results/clv_high_basin_blockdiag_20260317/collect/6bas_200k_dt0p005/paper_benchmark_summary.md)
  - [results/clv_high_basin_blockdiag_20260317/collect/8bas_200k_dt0p0025/paper_benchmark_summary.md](/home/mila/l/lia/skae/results/clv_high_basin_blockdiag_20260317/collect/8bas_200k_dt0p0025/paper_benchmark_summary.md)
- Fresh comparison outputs for the repaired MLP control rows also exist under
  the corresponding March 17 `compare/` directories.

March 19 `H3000` extension repair (`2026-03-19 16:11 EDT`):

- Root cause: the March 17 queue wrappers successfully launched training plus
  the usual `H100/H500/H1000` collect / compare jobs, but they never attached
  the repo's separate checkpoint-only `H1500/H2000/H2500/H3000` reevaluation
  stage. That stage is handled elsewhere by
  [scripts/paper_horizon_extension_eval_queue_20260311.sh](/home/mila/l/lia/skae/scripts/paper_horizon_extension_eval_queue_20260311.sh),
  not by the default training-time `full` evaluation profile.
- Fix applied:
  - added [scripts/queue_paper_horizon_extension_from_rows.sh](/home/mila/l/lia/skae/scripts/queue_paper_horizon_extension_from_rows.sh)
    to build a reevaluation manifest from collected `forecasting_rows.csv`
    files and submit the existing `H3000` queue driver
  - patched [scripts/queue_kuramoto_gs_blockdiag_fairness_20260317.sh](/home/mila/l/lia/skae/scripts/queue_kuramoto_gs_blockdiag_fairness_20260317.sh),
    [scripts/queue_hopfield_gs_blockdiag_fairness_20260317.sh](/home/mila/l/lia/skae/scripts/queue_hopfield_gs_blockdiag_fairness_20260317.sh),
    [scripts/queue_clv_15seed_extension_20260317.sh](/home/mila/l/lia/skae/scripts/queue_clv_15seed_extension_20260317.sh),
    and [scripts/queue_clv_high_basin_blockdiag_20260317.sh](/home/mila/l/lia/skae/scripts/queue_clv_high_basin_blockdiag_20260317.sh)
    so future reruns automatically submit the missing long-horizon stage after
    collection.
- Repair queue launched from
  [results/generic_sparse_blockdiag_rerun_h3000_extension_20260319](/home/mila/l/lia/skae/results/generic_sparse_blockdiag_rerun_h3000_extension_20260319):
  - queue driver `8986054` completed `0:0`
  - eval array `8986055` completed `0:0`
  - refresh job `8986056` completed `0:0`
- Refreshed long-horizon results:
  - Kuramoto: `generic_sparse_blockdiag` is now best from `H1000` through
    `H3000` with medians `6.39 / 9.87 / 12.74 / 15.13 / 17.59`
  - Hopfield quarter-`dt`: `generic_sparse_blockdiag` remains negative through
    `H3000` (`322.81 / 653.30 / 997.21 / 1284.10 / 1515.62`), while
    `generic_sparse` stays best
  - corrected `4`-basin `competitive_lv`: `generic_sparse_blockdiag_200k`
    remains negative through `H3000`
    (`0.2016 / 0.2729 / 0.3384 / 0.3962 / 0.4457`)
  - high-basin fixed-system `competitive_lv`: only `8`-basin `dt=0.005`
    becomes a positive blockdiag case at `H1000-H3000`
    (`0.4253 / 0.4020 / 0.3961 / 0.3976 / 0.4024`); the other three settings
    remain negative versus `generic_sparse_200k`
