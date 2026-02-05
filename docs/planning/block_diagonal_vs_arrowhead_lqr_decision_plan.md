# Block-Diagonal vs Arrowhead for LQR Readiness

Date: February 5, 2026
Owner: SKAE experiments
Status: Implemented (execution-ready)

## 1) Objective

Make a final architecture decision between `block_diagonal` and `arrowhead` Koopman structure for downstream local LQR control.

The decision must be based on **control readiness**, not only basin separability:
- Can we recover stable local linear models?
- Can we reliably attach a controller to a discovered local regime?
- Does performance hold when the assumed number of regimes is wrong?

## 2) Constraints in Intended Setting

In deployment we do **not** know:
- the number of basins in advance,
- which trajectories belong to which basin.

Therefore:
- Ground-truth basin labels are for diagnostics only.
- Primary decision metrics must be label-free.
- Experiments must test robustness to basin-count misspecification.

## 3) Core Questions

1. Does `arrowhead` improve LQR readiness over `block_diagonal`, or only improve representation-level separation?
2. Are `arrowhead` gains caused by structure or by exclusivity losses?
3. Which structure is more robust across seeds, latent size, and basin-count misspecification?
4. Which structure yields better closed-loop behavior after local LQR synthesis?

## 4) Hypotheses (Falsifiable)

H1 (Structure vs regularizer):
- `arrowhead` without exclusivity (`AH-ISO`) will lose much of the low-capacity separation advantage seen in `arrowhead` with exclusivity.

H2 (Control readiness):
- Best `block_diagonal` arm will match or beat best `arrowhead` arm on DARE feasibility and closed-loop stability at `target_size >= 256`.

H3 (Robustness to unknown basin count):
- `block_diagonal` will degrade less than `arrowhead` under wrong `B_proxy`.

H4 (Potential counter-case):
- `arrowhead` may win at low capacity (`target_size=128`) in assignment consistency and short-horizon controllability proxies, but may not retain advantage at `target_size=512`.

## 5) Experimental Arms

All arms use:
- `--config lista_nonlinear`
- `--env lyapunov --lyapunov_dim 8 --lyapunov_num_basins 13 --lyapunov_extend_mode embed --lyapunov_points_mode fixed`
- Pairwise training (`--pairwise`)
- `num_steps=10000`, `batch_size=512`

### A. Block-diagonal controls (from current best results)

BD-C1 (simplest strong control):
- `--k_structure block_diagonal`
- `--target_size 256`
- `--sparsity_coeff 1.0`
- `--block_balance_loss kl_uniform`
- `--block_balance_weight 1.0`
- no top1 margin terms

BD-C2 (best seed-averaged separation control):
- BD-C1 +
- `--block_one_block_loss top1_margin`
- `--block_top1_margin 0.05`
- `--block_one_block_weight 0.3`

### B. Arrowhead arms

AH-ISO (isolation arm: structure-only):
- `--structured`
- `--target_size 256` (derived via `d_global + B_proxy*d_basin`)
- `--lambda_exclusivity 0.0`
- `--lambda_global 0.0`
- `--lambda_local 0.0`
- `--lambda_sparsity 1.0`
- `--excl_warmup_steps 0`

AH-PRAG (practical arrowhead arm):
- `--structured`
- `--lambda_exclusivity 0.05`
- `--lambda_sparsity 0.3`
- `--excl_warmup_steps 2000`
- keep `lambda_global`, `lambda_local` at currently used values or zero (lock before run)

## 6) Basin-Count Misspecification Design

For each structure arm, repeat with assumed regime count `B_proxy`:
- Lyapunov: `{8, 13, 20}`
- Duffing: `{1, 2, 4}` (secondary validation system)

For `block_diagonal`:
- set `k_block_size = target_size // B_proxy`

For `arrowhead`:
- set `num_basins = B_proxy`
- choose `(d_global, d_basin)` so `d_global + B_proxy * d_basin = target_size`

This explicitly tests the real constraint: unknown basin count.

## 7) Staged Run Plan

### Stage 0: Sanity + Repro Check

Purpose:
- Reconfirm BD-C1 and BD-C2 at `target_size=256` under current code.

Runs:
- 2 seeds each, Lyapunov only.

Exit criteria:
- Metrics within expected range from `docs/EXPERIMENTS.md`.

### Stage 1: Structure Effect Isolation

Purpose:
- Compare BD best control vs AH-ISO (no exclusivity confound).

Runs:
- Arms: BD-C1, BD-C2, AH-ISO
- `target_size=256`
- `B_proxy in {8, 13, 20}`
- 4 seeds each.

### Stage 2: Practical Head-to-Head

Purpose:
- Decide practical winner under best realistic training settings.

Runs:
- Select best BD arm from Stage 1 (`BD*`)
- Compare `BD*` vs `AH-PRAG`
- `target_size in {128, 256, 512}`
- `B_proxy in {8, 13, 20}`
- 8 seeds each.

### Stage 3: Secondary System Transfer

Purpose:
- Check whether winner generalizes beyond Lyapunov.

Runs:
- Same winner comparison on Duffing
- `target_size in {128, 256}`
- `B_proxy in {1, 2, 4}`
- 6 seeds each.

## 8) Metrics

### 8.1 Primary Decision Metrics (Label-Free)

M1. Local linear fit quality:
- Discover regimes from latent trajectories (unsupervised).
- Fit local linear model per regime.
- Report 1-step and H-step latent NRMSE.

M2. LQR feasibility rate:
- Fraction of regimes where DARE solve succeeds.

M3. Closed-loop stability rate:
- Fraction of regimes with spectral radius `rho(A - B K_lqr) < 1`.

M4. Closed-loop improvement:
- Relative reduction in finite-horizon quadratic cost vs open loop.
- Recovery success from perturbations within regime neighborhoods.

M5. Robustness:
- Degradation across seeds and `B_proxy`.
- Report mean, std, and worst-decile performance.

### 8.2 Secondary Diagnostics

- Cosine separation and support uniqueness.
- Per-structure spectral radius summaries.
- Basin-to-block concentration (diagnostic only).
- Ground-truth basin assignment metrics (diagnostic only).

## 9) Evaluation Protocol (Per Run)

1. Train checkpoint (`tools/train.py`).
2. Run standard checkpoint eval (`tools/evaluate_checkpoints.py`).
3. Run spectral analysis (`tools/analyze_k_eigenvalues.py`).
4. Run support + cosine diagnostics (`tools/evaluate_support_uniqueness.py`, `tools/diagnose_cosine_separation.py`).
5. Run LQR-readiness evaluator (new script):
   - unsupervised regime discovery in latent space,
   - local `(A, B)` identification per regime,
   - DARE solve + closed-loop rollout,
   - save per-regime and aggregate metrics.

## 10) Statistical Plan

- Unit of replication: random seed.
- Report mean +/- 95% bootstrap CI per arm.
- Pairwise comparison between finalist arms:
  - paired bootstrap on seed-matched runs,
  - effect size + CI, not p-value only.
- Predefine failure handling:
  - if run diverges/non-finite, count as failure in primary metrics.

## 11) Pre-Registered Decision Rule

Choose winner by lexicographic rule:

1. Higher M2 (LQR feasibility rate) with CI excluding tie.
2. If tied, higher M3 (closed-loop stability rate).
3. If tied, better M4 (closed-loop cost reduction).
4. If tied, lower sensitivity to `B_proxy` misspecification (M5).
5. Secondary metrics used only for interpretation.

Minimum practical threshold to declare winner:
- At least +10% absolute gain in M2 or M3, with CI excluding 0.

If no arm passes threshold:
- Decision = "no clear winner", keep simpler arm (`block_diagonal`) and prioritize explicit spectral constraints / local-model improvements.

## 12) Expected Outcome Patterns (Hypothetical)

Pattern A (block-diagonal wins):
- Similar separability to arrowhead.
- Better DARE success and closed-loop stability.
- Smaller degradation when `B_proxy` is wrong.

Pattern B (arrowhead wins):
- Better regime consistency and local linear fit.
- Equal-or-better DARE/stability across all `target_size`.
- Robust under `B_proxy` mismatch.

Pattern C (split result):
- Arrowhead better on representation metrics, block-diagonal better on control metrics.
- Final decision still favors control winner for LQR objective.

## 13) Risks and Mitigations

Risk: Local `(A,B)` identification is noisy.
- Mitigation: regularized identification, minimum-sample threshold per regime, shrinkage checks.

Risk: Unsupervised regime discovery unstable.
- Mitigation: evaluate multiple clustering settings and keep a fixed pre-registered default.

Risk: Spectral instability dominates all arms.
- Mitigation: include failure as primary outcome; if widespread, branch to spectral-constrained rerun.

## 14) Deliverables

1. `results/<sweep>/summary_decision_table.csv`
2. `results/<sweep>/lqr_readiness_summary.json`
3. `results/<sweep>/final_decision.md` with:
- winner/undecided,
- primary metric table,
- robustness table across `B_proxy`,
- short interpretation.

## 15) Immediate Execution Checklist

1. Freeze exact `AH-PRAG` `(lambda_global, lambda_local)` values.
2. Launch Stage 0 runs and verify reproducibility.
3. Run Stage 1 and pick `BD*`.
4. Launch Stage 2 as main decision stage.
5. Only after Stage 2, run Stage 3 transfer check.

## 16) Implementation Notes (February 5, 2026)

The plan is now wired into runnable scripts/tools:

- LQR-readiness evaluator (label-free): `tools/evaluate_lqr_readiness.py`
- Sweep runner (single trial): `scripts/run_lqr_decision_trial.sh`
- Stage scripts:
  - `scripts/sweep_lqr_decision_stage0.sh`
  - `scripts/sweep_lqr_decision_stage1.sh`
  - `scripts/sweep_lqr_decision_stage2.sh`
  - `scripts/sweep_lqr_decision_stage3.sh`
- Aggregation + decision outputs:
  - `tools/collect_lqr_decision_results.py`

`AH-PRAG` lock chosen for reproducibility:
- `lambda_global = 1e-4`
- `lambda_local = 1e-3`

These values match the currently used structured defaults in this repository.
