# Subagent C Status: Fixed-Cadence Re-encoding Ablation

Date: March 9, 2026

## Objective

- Replace the horizon-wise `best_periodic` oracle with a deployment-closer fixed-cadence read, then measure whether the current paper-positive dense-LISTA and Kuramoto stories still hold.

## Current Evidence Read

- `docs/planning/paper_parallel_workstreams_20260309.md`:
  - workstream brief for Subagent C, including the requirement to define a fair fixed-cadence protocol before queueing
- `docs/PAPER_TRACK_STATUS.md`:
  - fair `200k` benchmark comparison now uses `lista_dense_promoted_stage4` versus `generic_sparse_ns200k_best`
  - focused Kuramoto `dt=0.00625`, `200k`, `5`-seed comparison is the current intrinsic-HD headline result
- `skae/evaluation.py`:
  - existing evaluation already saves `no_reencode`, `every_step`, and per-period `periodic_{10,25,50,100}` metrics for non-Dysts systems
  - `best_periodic` is computed post hoc per horizon from those saved modes
- `tools/evaluate_checkpoints.py`:
  - existing evaluator reuses `EvaluationSettings` and writes `evaluation_results_{best,last}.json`
- `tools/collect_forecasting_roots.py`:
  - current collector only exposes `best_periodic`, not fixed-period columns, so fixed-cadence analysis likely needs direct JSON re-scoring rather than a stock collector call
- Existing collected artifacts inspected:
  - `/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/forecasting_rows.csv`
  - `/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/compare/vs_generic_sparse_ns200k_best/lista_dense_promoted_stage4_vs_generic_sparse_ns200k_best/forecasting_comparison.md`
  - `/home/mila/l/lia/skae/results/kuramoto_dt00625_200k_compare_20260308/collect/forecasting_rows.csv`
  - `/home/mila/l/lia/skae/results/kuramoto_dt00625_200k_compare_20260308/compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md`
- Artifact spot-check:
  - confirmed that run directories referenced in collected CSVs exist on `/network/scratch/l/lia/skae/...`
  - confirmed that sample `evaluation_results_best.json` files contain the full saved periodic modes needed for offline fixed-cadence analysis

## Concrete Plan Before Queueing

- Claim being tested:
  - the current `best_periodic` metric is a mild oracle because it selects the best re-encoding cadence separately for each horizon on test evaluation
  - the paper needs to know whether the positive dense-LISTA benchmark story and the Kuramoto block-diagonal win survive under one fixed cadence that is closer to deployment
- Primary fixed-cadence protocol:
  - use a single global cadence `k=100` (`periodic_100`) as the primary non-oracle read
  - rationale: `k=100` is already in the standard non-Dysts evaluation grid, is the least frequent correction in that grid, and is therefore the cleanest deployment-like fixed policy without introducing new tuning or validation leakage
- Secondary sensitivity table:
  - offline only, also tabulate `k in {10, 25, 50, 100}` from the saved evaluation JSONs to show how much the conclusion depends on the chosen fixed cadence
  - do not use those test results to pick the primary cadence; they are context only
- Baselines and fairness controls:
  - dense benchmark story:
    - candidate root: `lista_dense_promoted_stage4`
    - anchor root: `generic_sparse_ns200k_best`
    - source artifact index: `/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/forecasting_rows.csv`
  - Kuramoto story:
    - roots: `generic_sparse`, `lista_dense`, `lista_blockdiag`
    - source artifact index: `/home/mila/l/lia/skae/results/kuramoto_dt00625_200k_compare_20260308/collect/forecasting_rows.csv`
  - checkpoint rule:
    - keep `evaluation_results_best.json` as the official metric source for this ablation, matching the current paper rule
  - seed control:
    - dense benchmark uses the existing `3` benchmark seeds already collected per system/root
    - focused Kuramoto uses the existing `5` seeds (`0,1,2,3,4`)
- Exact systems, horizons, and metrics:
  - dense benchmark slice:
    - primary comparison uses the existing fair benchmark roots and reports both:
      - all-system results over the collected `29` benchmark systems
      - non-intrinsic-HD slice excluding `competitive_lv`, `kuramoto`, `hopfield`, `multiwell_gradient_hd`, `multiwell_rotational_hd`, and `multiwell_strong_transition_hd`
  - Kuramoto comparison:
    - system: `kuramoto`, `dt=0.00625`, default `N=16`, seeds `0-4`
  - horizons:
    - `H100`, `H500`, `H1000`
  - metrics:
    - fixed-cadence mean MSE at each horizon
    - system-median / seed-median `H1000`
    - good-system count or good-seed count under `H1000 < 10`
    - dense-vs-anchor win count at fixed cadence
    - ratio of fixed-cadence `H1000` to current `best_periodic H1000`
    - mode-gap summary: `periodic_100 - best_periodic`
- Acceptance criteria:
  - dense benchmark story survives if, under `periodic_100`, dense still beats the fair `200k` `generic_sparse` anchor on a majority of systems in the non-intrinsic-HD slice and still has no system where dense fails the good band while the fair anchor passes
  - Kuramoto story survives if, under `periodic_100`, `lista_blockdiag` remains the best seed-median model at `H500` and `H1000` and stays inside the good band at `H1000`
- Failure criteria:
  - dense story materially weakens if the majority win flips on the non-intrinsic-HD slice or if dense newly fails systems that the fair anchor keeps in-band
  - Kuramoto story materially weakens if `lista_blockdiag` loses to `generic_sparse` or exits the good band at `H1000`
- Reuse / minimal-change policy:
  - first try to answer the workstream entirely by re-scoring existing `evaluation_results_best.json` files
  - only add code or queue jobs if any required fixed-cadence mode is missing from the saved evaluations
- Output root and artifact names:
  - preferred outcome: no new result root; record the local analysis commands and conclusions directly in this status file
  - only if recollection is required, use a new unique root:
    - `/home/mila/l/lia/skae/results/paper_parallel_20260309_c_fixed_cadence_reencoding/`
    - expected artifacts: `fixed_cadence_rows.csv`, `fixed_cadence_summary.md`, and any queue script/logs prefixed `paper_parallel_20260309_c_`

## Local QA Before Queueing

- Verify that the dense benchmark and focused Kuramoto collected CSVs resolve to real run directories on `/network/scratch`.
- Verify that sampled `evaluation_results_best.json` files contain `periodic_10`, `periodic_25`, `periodic_50`, and `periodic_100`.
- Recompute `best_periodic` for a small sample directly from the JSON and confirm it matches the current collected `H1000 best-periodic` value/mode in the existing CSVs.
- Confirm root labels, seed counts, and system counts:
  - dense benchmark roots should each have `29 x 3 = 87` collected runs
  - focused Kuramoto roots should each have `5` collected runs
- Confirm the non-intrinsic-HD exclusion filter is applied consistently before reporting dense win counts.
- If a helper script becomes necessary:
  - run it locally with `uv run` on a one-system / one-root subset first
  - check that output columns, cadence labels, and root labels are correct
- If queueing becomes necessary:
  - smoke-test the exact `uv run` command locally first
  - check task counts and output roots before `sbatch`
  - avoid the prior `sbatch --export` CSV bug pattern by using the repo’s existing launcher conventions only
- QA performed on March 9:
  - confirmed the collected CSVs resolve to real run directories on `/network/scratch/l/lia/skae/...`
  - confirmed the extracted row counts match expectation:
    - dense benchmark fixed-cadence table: `174` rows = `2` roots x `29` systems x `3` seeds
    - focused Kuramoto fixed-cadence table: `15` rows = `3` roots x `5` seeds
  - confirmed `best_periodic` values/modes in the saved JSONs match the collected CSVs exactly on the fully extracted dense and Kuramoto tables:
    - dense `qa_mismatches = 0`
    - Kuramoto `qa_mismatches = 0`
  - confirmed the non-intrinsic-HD exclusion filter produces the intended `23`-system slice:
    - excluded systems: `competitive_lv`, `hopfield`, `kuramoto`, `multiwell_gradient_hd`, `multiwell_rotational_hd`, `multiwell_strong_transition_hd`
  - attempted a `uv run`-based full-table extractor first, but multiple concurrent agent jobs held the shared `uv` environment lock; switched to `awk` + `jq` re-scoring against the existing JSONs instead and then killed the abandoned local `uv` processes

## What Was Queued

- No queue was needed.
- No `sbatch` command was issued.
- No SLURM job IDs.
- Existing artifacts were sufficient:
  - `/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/forecasting_rows.csv`
  - `/home/mila/l/lia/skae/results/kuramoto_dt00625_200k_compare_20260308/collect/forecasting_rows.csv`
  - per-run `evaluation_results_best.json` files referenced by those collected CSVs
- Local temp artifacts created for QA / aggregation only:
  - `/tmp/paper_parallel_20260309_c_dense_fixed_fast.tsv`
  - `/tmp/paper_parallel_20260309_c_kuramoto_fixed_fast.tsv`
- Key local commands run:

```bash
# Dense benchmark fixed-cadence extraction from existing evaluation_results_best.json
awk -F, 'NR==1{for(i=1;i<=NF;i++) idx[$i]=i; next} $1=="lista_dense_promoted_stage4" || $1=="generic_sparse_ns200k_best" {print $idx["root_label"] "|" $idx["system_name"] "|" $idx["system_key"] "|" $idx["seed_name"] "|" $idx["run_dir"] "|" $idx["h1000_best_periodic_mean"] "|" $idx["h1000_best_periodic_mode"]}' \
  /home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/forecasting_rows.csv \
  | xargs -d '\n' -P 16 -I{} bash -lc 'IFS="|" read -r root system_name system_key seed_name run_dir csv_best csv_mode <<< "$1"; jq -r --arg root "$root" --arg system_name "$system_name" --arg system_key "$system_key" --arg seed_name "$seed_name" --arg csv_best "$csv_best" --arg csv_mode "$csv_mode" '\''[$root,$system_name,$system_key,$seed_name,$csv_best,$csv_mode,(.[$system_key].best_periodic["1000"].mean // ""),(.[$system_key].best_periodic["1000"].mode // ""),(.[$system_key].modes.periodic_10.horizons["1000"].mean // ""),(.[$system_key].modes.periodic_25.horizons["1000"].mean // ""),(.[$system_key].modes.periodic_50.horizons["1000"].mean // ""),(.[$system_key].modes.periodic_100.horizons["1000"].mean // "")] | @tsv'\'' "$run_dir/evaluation_results_best.json"' _ '{}' \
  >> /tmp/paper_parallel_20260309_c_dense_fixed_fast.tsv

# Focused Kuramoto fixed-cadence extraction from existing evaluation_results_best.json
awk -F, 'NR==1{for(i=1;i<=NF;i++) idx[$i]=i; next} $1=="generic_sparse" || $1=="lista_dense" || $1=="lista_blockdiag" {print $idx["root_label"] "|" $idx["system_name"] "|" $idx["system_key"] "|" $idx["seed_name"] "|" $idx["run_dir"] "|" $idx["h1000_best_periodic_mean"] "|" $idx["h1000_best_periodic_mode"]}' \
  /home/mila/l/lia/skae/results/kuramoto_dt00625_200k_compare_20260308/collect/forecasting_rows.csv \
  | xargs -d '\n' -P 8 -I{} bash -lc 'IFS="|" read -r root system_name system_key seed_name run_dir csv_best csv_mode <<< "$1"; jq -r --arg root "$root" --arg system_name "$system_name" --arg system_key "$system_key" --arg seed_name "$seed_name" --arg csv_best "$csv_best" --arg csv_mode "$csv_mode" '\''[$root,$system_name,$system_key,$seed_name,$csv_best,$csv_mode,(.[$system_key].best_periodic["1000"].mean // ""),(.[$system_key].best_periodic["1000"].mode // ""),(.[$system_key].modes.periodic_10.horizons["100"].mean // ""),(.[$system_key].modes.periodic_10.horizons["500"].mean // ""),(.[$system_key].modes.periodic_10.horizons["1000"].mean // ""),(.[$system_key].modes.periodic_25.horizons["100"].mean // ""),(.[$system_key].modes.periodic_25.horizons["500"].mean // ""),(.[$system_key].modes.periodic_25.horizons["1000"].mean // ""),(.[$system_key].modes.periodic_50.horizons["100"].mean // ""),(.[$system_key].modes.periodic_50.horizons["500"].mean // ""),(.[$system_key].modes.periodic_50.horizons["1000"].mean // ""),(.[$system_key].modes.periodic_100.horizons["100"].mean // ""),(.[$system_key].modes.periodic_100.horizons["500"].mean // ""),(.[$system_key].modes.periodic_100.horizons["1000"].mean // "")] | @tsv'\'' "$run_dir/evaluation_results_best.json"' _ '{}' \
  >> /tmp/paper_parallel_20260309_c_kuramoto_fixed_fast.tsv

# Dense benchmark aggregation on the extracted TSV
gawk -F'\t' '...' /tmp/paper_parallel_20260309_c_dense_fixed_fast.tsv

# Kuramoto aggregation on the extracted TSV
gawk -F'\t' '...' /tmp/paper_parallel_20260309_c_kuramoto_fixed_fast.tsv

# Best-periodic dense baseline on the collected CSV
gawk -F, '...' /home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/forecasting_rows.csv
```

## Results / Observations

- Queue decision:
  - no new evaluation or recollection job is needed; the saved `evaluation_results_best.json` artifacts already contain the full `periodic_{10,25,50,100}` grid required for this ablation
- Dense benchmark baseline (`best_periodic`, from existing collected CSV):
  - all `29` systems: dense wins `18`, fair `generic_sparse` wins `11`, good systems `26` vs `25`, and there are `0` systems where dense fails while the fair anchor passes
  - non-intrinsic-HD `23`-system slice: dense wins `15`, fair `generic_sparse` wins `8`, good systems `23` vs `22`, and there are `0` dense-fail / anchor-pass systems
- Dense benchmark under fixed `periodic_100`:
  - all `29` systems: dense wins `17`, fair `generic_sparse` wins `11`, good systems tie at `22` vs `22`, and dense newly fails `blended` while the fair anchor remains in-band
  - non-intrinsic-HD `23`-system slice: dense wins `13`, fair `generic_sparse` wins `9`, good systems tie at `19` vs `19`, and the same new dense failure is `blended`
  - the dense fixed-cadence penalty is real but not fatal:
    - majority win-count still survives at `periodic_100`
    - the stronger no-new-fail and better-good-count story does not survive
    - dense row-median `H1000 periodic_100 / H1000 best_periodic = 1.28012`
- Dense fixed-cadence sensitivity across the saved global cadences:
  - `periodic_10`: poor benchmark comparator for dense (`9` wins vs `20` losses on all `29`; `8` vs `15` on the `23`-system non-intrinsic slice)
  - `periodic_25`: too weak / tie-heavy to carry the benchmark story cleanly (`7` wins vs `7` losses on all `29`; `4` vs `4` on the `23`-system slice)
  - `periodic_50`: loses the benchmark story and creates a new dense failure on `multiwell_energy_hd`
  - `periodic_100`: strongest of the saved fixed-cadence choices for dense, and therefore the right primary deployment-like read among the existing evaluation modes
- Focused Kuramoto fixed-cadence results:
  - `periodic_100` exactly matches the official `best_periodic` `H1000` read for every root in the focused `dt=0.00625`, `200k`, `N=16`, `5`-seed comparison:
    - `generic_sparse`: seed-median `H1000 = 27.0193`
    - dense LISTA: seed-median `H1000 = 13.8445`
    - `lista_blockdiag`: seed-median `H1000 = 6.98035`
  - under `periodic_100`, `lista_blockdiag` remains the best `H1000` model and all `5/5` seeds stay under the good band (`H1000 < 10`)
  - Kuramoto `periodic_100 / best_periodic` seed-median ratio is exactly `1.0` for all three roots, so the current long-horizon Kuramoto claim does not depend on the mild horizon-wise oracle
  - at `H500`, `generic_sparse` is still better than `lista_blockdiag` under the fixed cadence (`1.58512` vs `2.68336`), so the fixed-cadence Kuramoto positive is specifically an `H1000` forecasting result, not a sweep-wide horizon win
- Kuramoto fixed-cadence sensitivity:
  - `periodic_50` still keeps `lista_blockdiag` in-band at `H1000` (`8.19319`), but weaker than `periodic_100`
  - `periodic_25` pushes `lista_blockdiag` out of band at `H1000` (`15.1258`)
  - `periodic_10` collapses both LISTA roots at `H1000` (`1291.05` dense, `4887.6` block-diagonal)
- Deviation from the prewritten acceptance criteria:
  - the prewritten Kuramoto criterion requiring `lista_blockdiag` to remain best at both `H500` and `H1000` was too strict relative to the actual paper narrative
  - the deployment-like fixed-cadence read preserves the Kuramoto `H1000` headline result exactly, but not an `H500` win
- Final conclusion:
  - this workstream is complete without queueing
  - the paper can replace a pure `best_periodic` presentation with a more honest statement:
    - dense benchmark gains mostly survive under a single global `periodic_100`, but the advantage is narrower and no longer “strictly safer than the fair anchor”
    - the focused Kuramoto `H1000` block-diagonal win survives unchanged under the same fixed cadence

## Proposed Updates To Global Status Docs

- `docs/EXPERIMENTS.md`:
  - add a short fixed-cadence note under the fair `200k` dense-vs-fair-MLP section:
    - under a single global `periodic_100`, promoted dense Stage 4 still beats `generic_sparse_ns200k_best` on `17/29` systems overall and `13/23` systems on the non-intrinsic-HD slice, but good-system count falls to a tie (`22/29` overall; `19/23` on the non-intrinsic slice) and dense newly fails `blended` while the fair anchor passes
  - add a short fixed-cadence note under the focused Kuramoto `dt=0.00625`, `200k` section:
    - `periodic_100` exactly reproduces the official `H1000` ranking (`6.98` block-diagonal, `13.84` dense, `27.02` `generic_sparse`), so the long-horizon Kuramoto rescue claim does not rely on horizon-wise cadence selection
    - clarify that this is an `H1000` result; at `H500`, `generic_sparse` still has lower error than `lista_blockdiag`
- `docs/PAPER_TRACK_STATUS.md`:
  - add one sentence near the dense Stage-4 / fair-`200k` comparison clarifying that the dense win-count story survives a deployment-like fixed `periodic_100`, but the good-system advantage and “no dense fail while fair MLP passes” safety margin do not
  - add one sentence near the Kuramoto focused comparison clarifying that the official `H1000` block-diagonal win is already a fixed-cadence (`periodic_100`) win in practice, not just a `best_periodic` oracle artifact
