# Subagent E Status: Kuramoto Robustness Beyond Default

Date: March 9, 2026

## Objective

Test whether the current Kuramoto block-diagonal LISTA win is specific to the exact default environment (`N=16`, ring, identical frequencies, `dt=0.00625`, `200k`) or survives a mild regime shift that keeps the task recognizably Kuramoto.

## Current Evidence Read

- Read the workstream spec in `docs/planning/paper_parallel_workstreams_20260309.md`.
- Read `docs/planning/high_dim_benchmarks_plan.md` and `docs/planning/kuramoto_recovery_plan.md`.
- Read the current paper-track summaries in `docs/EXPERIMENTS.md` and `docs/PAPER_TRACK_STATUS.md`.
- Audited the environment/config wiring in `skae/config.py`, `skae/data.py`, `tools/train.py`, `tools/build_paper_benchmark_tasks.py`, `tools/build_kuramoto_dimension_sweep_tasks.py`, and `scripts/run_paper_benchmark_array.sh`.
- Read the completed focused comparison and scaling artifacts:
  - `results/kuramoto_dt00625_200k_compare_20260308/collect/forecasting_summary.md`
  - `results/kuramoto_n32_dt00625_200k_confirm_20260309/forecasting_summary.md`
  - `results/kuramoto_dimension_sweep_dt00625_200k_20260309/collect/kuramoto_dimension_summary.md`
- Current decision-grade baseline:
  - default setting: `N=16`, ring, identical frequencies, `dt=0.00625`, `200k`
  - `generic_sparse`: seed-median `H1000` best-periodic `27.02`
  - dense LISTA: seed-median `H1000` best-periodic `13.84`
  - `lista_blockdiag`: seed-median `H1000` best-periodic `6.98`
- Current scaling read already shows the positive result is not only a single dimension point: `lista_blockdiag` remains good through `N=32`, but that is still the same environment family and does not answer the regime-robustness question.
- The repo already supports the preferred robustness lever in the environment itself:
  - `KURAMOTO.OMEGA_MODE in {identical, uniform_spread, random}`
  - `KURAMOTO.TOPOLOGY in {ring, all_to_all}`
- Existing task builders and array runners do not yet expose the Kuramoto frequency/topology knobs, so a minimal wiring patch is required before queueing.

## Concrete Plan Before Queueing

### Claim being tested

The paper-facing claim is not “block-diagonal wins only on the exact default Kuramoto setting,” but “the smaller-`dt`, longer-training block-diagonal LISTA rescue survives a mild change in regime.” The cheapest honest test is mild frequency heterogeneity with everything else fixed.

### Chosen robustness lever

- Primary lever: `OMEGA_MODE=uniform_spread`
- Keep:
  - `OMEGA_SPREAD=0.5` (existing environment default; mild perturbation relative to `K_COUPLING=4.0`)
  - `TOPOLOGY=ring`
  - `N=16`
  - `dt=0.00625`
  - `num_steps=200000`
  - `sequence_length=8`
  - `target_size=256`
- Do not include a topology change in this first pass. `all_to_all` is a valid future fallback, but it is a second lever and is not needed for the minimal paper question.

### Baselines and fairness controls

- Queue only:
  - `generic_sparse` as the MLP anchor
  - `lista_blockdiag` as the candidate
- Omit dense LISTA in the first pass to keep the queue minimal. Bring it back only if the `generic_sparse` vs `lista_blockdiag` result is ambiguous or if the paper narrative later needs a three-way robustness table.
- Keep the winning/default recipes fixed:
  - `generic_sparse`: same recipe used in the focused default comparison
  - `lista_blockdiag`: same promoted recipe used in the focused default comparison (`lista_alpha=0.15`, `lista_num_loops=1`, `k_block_size=16`, `lista_final_op=relu`, `sparsity_coeff=0.006`)

### Exact systems, seeds, horizons, and metrics

- System:
  - `kuramoto`
- Environment:
  - `NUM_OSCILLATORS=16`
  - `TOPOLOGY=ring`
  - `OMEGA_MODE=uniform_spread`
  - `OMEGA_SPREAD=0.5`
  - `DT=0.00625`
- Seeds:
  - `0,1,2,3,4`
- Horizons:
  - `H100`, `H500`, `H1000`
- Primary metric:
  - seed-median `H1000` best-periodic
- Secondary metrics:
  - all-seeds-good at `H1000`
  - worst-seed `H1000` best-periodic
  - `H100` and `H500` best-periodic
  - every-step vs best-periodic gap for context

### Promotion and failure criteria

- Promotion / positive robustness result:
  - `lista_blockdiag` still beats `generic_sparse` on seed-median `H1000` best-periodic, and
  - `lista_blockdiag` remains inside the good-forecast band (`< 10`) or degrades only modestly while preserving a clear edge over `generic_sparse`, and
  - no catastrophic block-diagonal tail appears (`worst-seed H1000 << 1e3`; practical target: `< 100`)
- Ambiguous result:
  - both models remain good and close (`ratio` near `1.0`), or both degrade but neither clearly fails
  - action if this happens: consider a narrow dense-LISTA add-on or an `N=32` follow-up under the same heterogeneity setting
- Failure / non-robust result:
  - `lista_blockdiag` loses to `generic_sparse` on seed-median `H1000`, or
  - `lista_blockdiag` leaves the good band with obvious tail instability while `generic_sparse` stays competitive

### Output roots and artifact names

- Scratch root:
  - `/network/scratch/l/lia/skae/paper_parallel_20260309_e_kuramoto_uniformspread_n16_dt00625_200k`
- Repo results root:
  - `results/paper_parallel_20260309_e_kuramoto_uniformspread_n16_dt00625_200k`
- Planned artifacts:
  - `task_tables/paper_parallel_20260309_e_kuramoto_robustness.tsv`
  - `task_tables/paper_parallel_20260309_e_kuramoto_robustness_manifest.json`
  - `root_specs/paper_parallel_20260309_e_kuramoto_robustness_roots.txt`
  - `collect/forecasting_rows.csv`
  - `collect/forecasting_summary.md`
  - `compare/lista_blockdiag_uniform_spread_vs_generic_sparse_uniform_spread/forecasting_comparison.md`

### Minimal code/script plan

- Add minimal Kuramoto CLI overrides to `tools/train.py`:
  - topology
  - omega mode
  - omega spread
- Add a unique task builder:
  - `tools/paper_parallel_20260309_e_build_kuramoto_robustness_tasks.py`
- Add a unique queue launcher:
  - `scripts/paper_parallel_20260309_e_queue_kuramoto_robustness.sh`
- Reuse existing collection and comparison scripts after generating a custom task table and root-spec file.

## Local QA Before Queueing

Checklist to complete before submission:

- [ ] Confirm the new CLI args in `tools/train.py` correctly override `cfg.ENV.KURAMOTO`.
- [ ] Confirm the custom task builder writes the expected rows and metadata.
- [ ] Confirm task count is exactly `10` (`2` models x `5` seeds).
- [ ] Confirm the task TSV contains explicit Kuramoto regime columns and no accidental CSV-in-`sbatch --export` bug pattern.
- [ ] Confirm the custom queue script uses `uv run` for Python entry points and `sbatch` for submission.
- [ ] Confirm log/output directories are regime-specific and cannot collide with the existing default-setting runs.
- [ ] Smoke-test the task table locally.
- [ ] Smoke-test the train command locally on a single task in dry-run mode.
- [ ] Run a short local instantiation check for the heterogenous Kuramoto environment before queueing.
- [ ] Re-read the generated root labels and compare target names before submission.

## What Was Queued

- Submitted the Subagent E Kuramoto robustness chain from the subagent workspace:
  - array job `8914740`
    - job name: `e_kuramoto`
    - task range: `0-9`
  - dependent collect job `8914741`
    - job name: `collect_paper`
    - dependency: `afterany:8914740_*`
  - dependent compare job `8914742`
    - job name: `compare_paper`
    - dependency: `afterany:8914741`
- Current scheduler state when checked:
  - `8914740` pending in `long`
  - `8914741` pending on dependency in `long-cpu`
  - `8914742` pending on dependency in `long-cpu`

## Results / Observations

- The robustness workstream is now past planning and into queued execution.

## Proposed Updates To Global Status Docs
ators`, `kuramoto_topology`, `kuramoto_omega_mode`, `kuramoto_omega_spread`
    - expected phase label and model recipes present
- Runner dry-run:
  - `module load cuda/12.6.0 && module load cuda/12.6.0/cudnn/9.3 && SLURM_ARRAY_TASK_ID=0 DRY_RUN=1 TASK_TSV=/tmp/paper_parallel_20260309_e_qa.UcIDNm/tasks.tsv BASE_OUT=/tmp/paper_parallel_20260309_e_base bash scripts/paper_parallel_20260309_e_run_kuramoto_robustness_array.sh`
  - Verified:
    - generated train command includes `--kuramoto_num_oscillators 16 --kuramoto_topology ring --kuramoto_omega_mode uniform_spread --kuramoto_omega_spread 0.5`
    - generated `LOG_DIR` is regime-specific:
      - `/tmp/paper_parallel_20260309_e_base/paper_parallel_20260309_e_kuramoto_uniformspread_n16_dt00625_200k/generic_sparse/n_16/topo_ring/omega_uniform_spread/spread_0p5/kuramoto/dt_0p00625/seed_0`
- Root-label / compare-target check:
  - `results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/root_specs/paper_parallel_20260309_e_kuramoto_robustness_roots.txt`
  - Verified labels:
    - `generic_sparse_uniform_spread`
    - `lista_blockdiag_uniform_spread`
- Local runtime caveat:
  - direct login-shell `torch` import remained blocked by missing NVIDIA runtime libs (`libcusparseLt.so.0`) even after loading `cuda/12.6.0` and `cuda/12.6.0/cudnn/9.3`
  - this prevented a full local environment-instantiation smoke test through `make_env(...)`
  - I still queued because:
    - the builder/runner/path wiring was locally validated
    - the actual training path is meant to run on SLURM compute nodes, and prior Kuramoto jobs in this repo have succeeded there

## What Was Queued

- Queue command:
  - `sbatch scripts/paper_parallel_20260309_e_queue_kuramoto_robustness.sh`
- Queue launcher job:
  - `8914718`
- Downstream SLURM chain from `slurm-8914718.out`:
  - array: `8914740_[0-9]`
  - collector: `8914741`
  - comparison: `8914742`
- Current queue state immediately after launch:
  - `8914740_[0-9]` pending
  - `8914741` pending on dependency
  - `8914742` pending on dependency
- Exact scripts used:
  - `scripts/paper_parallel_20260309_e_queue_kuramoto_robustness.sh`
  - `scripts/paper_parallel_20260309_e_run_kuramoto_robustness_array.sh`
  - `scripts/collect_paper_benchmark.sh`
  - `scripts/compare_paper_benchmark.sh`
- Exact output paths:
  - scratch root:
    - `/network/scratch/l/lia/skae/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309`
  - repo results root:
    - `results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309`
  - task table:
    - `results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/task_tables/paper_parallel_20260309_e_kuramoto_robustness.tsv`
  - manifest:
    - `results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/task_tables/paper_parallel_20260309_e_kuramoto_robustness_manifest.json`
  - root specs:
    - `results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/root_specs/paper_parallel_20260309_e_kuramoto_robustness_roots.txt`
- Exact queued settings:
  - models: `generic_sparse`, `lista_blockdiag`
  - seeds: `0,1,2,3,4`
  - environment: `kuramoto`, `N=16`, `topology=ring`, `omega_mode=uniform_spread`, `omega_spread=0.5`
  - training budget: `dt=0.00625`, `num_steps=200000`, `sequence_length=8`, `target_size=256`

## Results / Observations

- No experimental forecasting results yet; only the queue launcher has completed.
- Queue launcher successfully generated the task table and submitted the intended `10`-task array plus dependent collector/comparison jobs.
- Deviation from ideal local QA:
  - the final pre-queue environment-instantiation smoke test was blocked by the login-node GPU runtime, not by the task-table or queue wiring
  - no code deviation from the planned experiment recipe was needed

## Proposed Updates To Global Status Docs

- No immediate claim update is warranted before results are collected.
- Once the collector finishes, propose one of the following updates:
  - positive case:
    - `docs/EXPERIMENTS.md`: add a short entry that the `N=16`, `uniform_spread` Kuramoto follow-up preserves the block-diagonal advantage beyond the identical-frequency default
    - `docs/PAPER_TRACK_STATUS.md`: upgrade the Kuramoto narrative from “default-setting win with dimension scaling to `N=32`” to “default-setting win plus mild frequency-heterogeneity robustness”
  - negative case:
    - `docs/EXPERIMENTS.md`: add that the smaller-`dt` block-diagonal rescue is sensitive to mild frequency heterogeneity
    - `docs/PAPER_TRACK_STATUS.md`: qualify the current Kuramoto positive as strong but default-regime-specific rather than mild-regime-robust
- Independent of outcome, the queue itself can be logged later as:
  - “Subagent E queued the minimal Kuramoto robustness sweep under `results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309` with launcher `8914718`, array `8914740_[0-9]`, collector `8914741`, comparison `8914742`.”
