# LISTA Depth-First Forecasting Plan (Dysts + Multi-Well)

Date: February 19, 2026  
Owner: SKAE experiments  
Status: Queued (cache-gated execution in progress)

## 0) Execution Status (February 19, 2026)

1. Dysts cache prebuild matrix is running:
   - `8742755` (`scripts/prebuild_dysts_cache_matrix.sh`, array `0-95`).
2. Generic Sparse MLP benchmark chain is queued with dependency `afterok:8742755`:
   - `8743073`: `scripts/sweep_dysts_forecast_generic_sparse_ts256_cachebench.sh` (15 systems x 3 seeds = 45 runs).
   - `8743074`: `scripts/collect_dysts_forecast_generic_sparse_ts256_cachebench.sh` (`afterany:8743073`).
3. LISTA depth-first Phase-1 chain is queued with dependency `afterok:8742755`:
   - `8743075`: `scripts/sweep_lista_depth_phase0_smoke_23sys.sh` (4-system smoke).
   - `8743076`: `scripts/sweep_lista_depth_phase1_23sys.sh` (345 runs, `afterok:8743075`).
   - `8743077`: `scripts/collect_lista_depth_phase1_23sys.sh` (`afterany:8743076`).
   - `8743078`: `scripts/compare_lista_depth_phase1_23sys.sh` (`afterany:8743077`).

## 1) Objective

Tune LISTA to learn stable dynamics first, then recover sparsity/support structure.

Priority order:
1. Forecasting robustness and autonomous rollout stability.
2. Sparsity and basin-support uniqueness.

This is a deliberate strategy shift from sparsity-first to depth-first optimization.

## 2) Key Decisions Locked

1. Benchmark set: full 23 systems (15 dysts + 8 multi-well).
2. Training priority in Phase 1: forecasting only (support metrics are logged but non-gating).
3. Depth sweep first under low sparsity.
4. Low sparsity baseline for depth sweep:
   - `sparsity_coeff = 0.10`
   - `lista_alpha = 0.15`
5. Depth grid:
   - `lista_num_loops in {1, 2, 3, 5, 7}`

## 3) Constraints for Intended Deployment

In training/deployment we do not know basin labels or basin count.

Therefore:
1. Label-based basin metrics are benchmark diagnostics only.
2. Primary progress gates should be based on forecasting robustness and seed stability.
3. Basin-support alignment remains the long-term objective, but not the Phase-1 gate.

## 4) Canonical System Set (23)

### Dysts (15)
1. `dysts:Dadras`
2. `dysts:Duffing`
3. `dysts:QiChen`
4. `dysts:Sakarya`
5. `dysts:SprottTorus`
6. `dysts:Chua`
7. `dysts:MultiChua`
8. `dysts:DequanLi`
9. `dysts:LuChenCheng`
10. `dysts:SanUmSrisuchinwong`
11. `dysts:WangSun`
12. `dysts:ShimizuMorioka`
13. `dysts:LorenzCoupled`
14. `dysts:RikitakeDynamo`
15. `dysts:Hadley`

### Multi-well 2D (4)
16. `multiwell_gradient`
17. `multiwell_rotational`
18. `multiwell_energy`
19. `multiwell_strong_transition`

### Multi-well 8D lifted aliases (4)
20. `multiwell_gradient_hd`
21. `multiwell_rotational_hd`
22. `multiwell_energy_hd`
23. `multiwell_strong_transition_hd`

Seeds per configuration: `{0, 1, 2}`.

## 5) Fixed Training Protocol

Unless explicitly varied:
1. `--config lista_nonlinear`
2. `--target_size 256`
3. `--pairwise`
4. `--lista_final_op relu`
5. Dysts data handling matched to existing forecasting runs:
   - `--standardize`
   - `--dysts_ic_noise_scale 0.2`
   - `--dysts_native_cache`
   - `--dysts_cache_warmup 2000`

## 6) Phase Plan

### Phase 0: Smoke Validation

Purpose:
1. Verify training/evaluation pipeline and artifact shape before large sweeps.

Runs:
1. Systems: `dysts:Dadras`, `dysts:Chua`, `multiwell_gradient`, `multiwell_gradient_hd`.
2. One depth setting (`num_loops=3`) and one seed (`seed=0`).

Exit criteria:
1. Checkpoints produced.
2. `evaluation_results_best.json` produced.
3. Collector scripts ingest outputs (after utility fix in Section 9).

### Phase 1: Depth Sweep (Forecasting Gate Only)

Purpose:
1. Find depth that best supports robust long-horizon forecasting under low sparsity.

Sweep:
1. `lista_num_loops in {1,2,3,5,7}`
2. Fixed: `sparsity_coeff=0.10`, `lista_alpha=0.15`

Run count:
1. `5 depths x 23 systems x 3 seeds = 345 runs`.

Primary gate (hard):
1. Improve median `H1000 best-periodic`.
2. Reduce catastrophic systems (`H1000 >= 1000`).
3. Reduce dependence on `periodic_1`.
4. Improve seed robustness (`all-seeds-good` up, `any-seed-catastrophic` down).

Support metrics in Phase 1:
1. Logged only, not gating.

Output:
1. Select one winning depth (`depth_star`).

### Phase 2: Sparsity Tuning With Fixed Depth

Purpose:
1. Recover stronger support structure without losing Phase-1 forecasting gains.

Step 2A (coarse sparsity-coefficient sweep):
1. Fix `lista_alpha=0.15`, `lista_num_loops=depth_star`.
2. Sweep `sparsity_coeff in {0.05, 0.10, 0.20, 0.40, 0.80}`.

Step 2B (alpha sweep around best coefficient):
1. Fix `sparsity_coeff=sp_star` from Step 2A, `lista_num_loops=depth_star`.
2. Sweep `lista_alpha in {0.10, 0.15, 0.25, 0.35}`.

Step 2C (optional local joint refinement):
1. If needed, test a small neighborhood around `(depth_star, sp_star, alpha_star)`.

Gate (hard, re-balanced):
1. Forecasting must stay within acceptable degradation vs Phase-1 best.
2. Support metrics (CosSep/uniqueness/consistency) should improve relative to Phase-1 best.
3. No return of catastrophic heavy tails.

## 7) Metrics and Decision Rules

Forecasting metrics (primary):
1. `H1000 no_reencode`, `H1000 every_step`, `H1000 best_periodic`.
2. Best periodic mode distribution.
3. Good systems count with threshold `H1000 best-periodic < 10`.
4. Catastrophic systems count with threshold `H1000 best-periodic >= 1000`.

Seed-robustness metrics:
1. `all_seeds_good_systems`.
2. `any_seed_bad_systems`.
3. `any_seed_catastrophic_systems`.

Support diagnostics (secondary in Phase 1, primary-secondary in Phase 2):
1. Cosine separation metrics.
2. Support uniqueness and consistency.
3. Threshold sweeps only as secondary diagnostics.

Decision rule for `depth_star`:
1. Minimize catastrophic tails first.
2. Then maximize good-system count.
3. Then minimize median `H1000 best-periodic`.
4. Break ties by seed-robustness.

## 8) Output and Artifact Conventions

Suggested output root:
1. `/network/scratch/l/lia/skae/lista_depth_first_23sys`

Layout:
1. `${BASE_OUT}/phase1_depth/${DEPTH}/${SYSTEM}/seed_${SEED}/<timestamp>/...`
2. `${BASE_OUT}/phase2_sparsity/${CONFIG_TAG}/${SYSTEM}/seed_${SEED}/<timestamp>/...`

Required per run:
1. `config.json`
2. `checkpoint.pt`, `last.pt`
3. `evaluation_results_best.json`

## 9) Useful Pre-Implementation Utilities (Required for 23-System Collection)

Current gap:
1. `tools/collect_dysts_forecasting.py` currently extracts only keys beginning with `dysts:` and will skip `multiwell*`.

Required utility work before full collection:
1. Add a generalized collector for forecasting roots that accepts any system key (dysts and built-in).
2. Add a matching comparison script for candidate vs anchor roots on mixed system sets.
3. Keep current dysts-only tools for backward compatibility.

Suggested names:
1. `tools/collect_forecasting_roots.py`
2. `tools/compare_forecasting_roots.py`

## 10) Suggested Script Set (for implementation phase)

1. `scripts/sweep_lista_depth_phase1_23sys.sh`
2. `scripts/collect_lista_depth_phase1_23sys.sh`
3. `scripts/compare_lista_depth_phase1_23sys.sh`
4. `scripts/sweep_lista_sparsity_phase2_23sys.sh`
5. `scripts/collect_lista_sparsity_phase2_23sys.sh`
6. `scripts/compare_lista_sparsity_phase2_23sys.sh`

## 11) Command Templates

Single run template:

```bash
uv run python tools/train.py \
  --config lista_nonlinear \
  --env "${SYSTEM}" \
  --num_steps 10000 \
  --batch_size 256 \
  --target_size 256 \
  --reconst_coeff 0.5 \
  --pred_coeff 1.0 \
  --sparsity_coeff "${SPARSITY_COEFF}" \
  --lista_alpha "${LISTA_ALPHA}" \
  --lista_num_loops "${NUM_LOOPS}" \
  --lista_final_op relu \
  --pairwise \
  --standardize \
  --dysts_ic_noise_scale 0.2 \
  --dysts_native_cache \
  --dysts_cache_warmup 2000 \
  --seed "${SEED}" \
  --device cuda \
  --log_dir "${OUT_DIR}"
```

Checkpoint evaluation template:

```bash
uv run python tools/evaluate_checkpoints.py \
  --run_dir "${RUN_DIR}" \
  --system "${SYSTEM}" \
  --device cuda
```

## 12) Tracking Tables

### Phase-1 depth summary

| depth | runs_expected | runs_collected | good_systems | catastrophic_systems | median_h1000_best_periodic | periodic_1_count | all_seeds_good | any_seed_catastrophic | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 69 |  |  |  |  |  |  |  |  |
| 2 | 69 |  |  |  |  |  |  |  |  |
| 3 | 69 |  |  |  |  |  |  |  |  |
| 5 | 69 |  |  |  |  |  |  |  |  |
| 7 | 69 |  |  |  |  |  |  |  |  |

### Phase-2 sparsity summary

| config_tag | runs_expected | runs_collected | good_systems | catastrophic_systems | median_h1000_best_periodic | support_cosep | support_uniqueness | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| sp0.05_a0.15 | 69 |  |  |  |  |  |  |  |
| sp0.10_a0.15 | 69 |  |  |  |  |  |  |  |
| sp0.20_a0.15 | 69 |  |  |  |  |  |  |  |
| sp0.40_a0.15 | 69 |  |  |  |  |  |  |  |
| sp0.80_a0.15 | 69 |  |  |  |  |  |  |  |

## 13) Risks and Mitigations

Risk 1:
1. Depth sweep still dominated by catastrophic systems on a few hard dynamics.
Mitigation:
1. Keep per-system tail diagnostics and do not choose `depth_star` from median alone.

Risk 2:
1. Low sparsity improves forecasting but destroys support structure.
Mitigation:
1. Accept temporarily in Phase 1; Phase 2 explicitly restores support under forecasting constraints.

Risk 3:
1. Mixed-system collector tooling fails on multi-well.
Mitigation:
1. Implement Section 9 utility updates before full queue collection.

## 14) Definition of Done

1. Phase 1 completed with a selected `depth_star` on the 23-system benchmark.
2. Phase 2 completed with at least one sparsity setting that preserves Phase-1 robustness and improves support diagnostics.
3. Results summarized in `docs/EXPERIMENTS.md` using the standard reporting protocol.
