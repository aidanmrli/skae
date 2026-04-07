# GPT-5.4 Parallel Workstreams

Date: March 5, 2026

These three workstreams are intentionally independent and can be handed to separate GPT-5.4 subagents.

## Subagent A: HyperLISTA Stabilization

Objective:
- Turn Queue-5 from a negative result into a controlled recovery attempt, with code-path fixes before any new sweep.

Primary inputs:
- `docs/planning/hyperlista_stabilization_plan.md`
- `docs/EXPERIMENTS.md`
- `results/duffing_hyperlista_q05_adaptive_50k_20260304/`
- `skae/model.py`
- `skae/config.py`
- `tools/train.py`
- `tests/test_hyperlista.py`

Required tasks:
1. Audit the current HyperLISTA path against the stabilization plan.
2. Implement the minimum safe patch set:
   - positive `c_theta` parameterization
   - safe `pinv(D)` refresh
   - optional support-selection disable
   - optional sparsity-target penalty
3. Add focused tests for the new failure modes.
4. Prepare Queue-5b quick-sweep scripts, but do not launch without explicit sign-off.
5. Update `docs/EXPERIMENTS.md` with rationale and acceptance criteria.

Deliverables:
- code patch
- passing targeted tests
- Queue-5b sweep script(s)
- updated `docs/EXPERIMENTS.md`

Done criteria:
- at least one Queue-5b-ready arm is configured to target `sparsity_ratio ~ 0.7-0.9`
- HyperLISTA no longer depends on stale `pinv` caching or unconstrained threshold scaling
- the next sweep is small enough to fail cheaply

## Subagent B: Intrinsic-HD Result Audit and Kuramoto Recovery

Objective:
- Turn the March 5 intrinsic-HD baseline into a concrete Kuramoto recovery experiment, while keeping `hopfield` as a separate blocker.

Primary inputs:
- `docs/EXPERIMENTS.md`
- `docs/planning/high_dim_benchmarks_plan.md`
- `results/high_dim_benchmarks_plan_seq8_20260305/`
- `/network/scratch/l/lia/skae/high_dim_benchmarks_plan_seq8_20260305/`
- `scripts/sweep_high_dim_benchmarks_seq8.sh`
- `skae/data.py`
- `skae/evaluation.py`

Required tasks:
1. Summarize the current intrinsic-HD results with emphasis on what is solved vs blocked.
2. Diagnose Kuramoto failure mode using per-seed metrics, spectral radius, sparsity ratio, and rollout mode selection.
3. Prefer the cheapest plausible recovery first:
   - retune `lista_blockdiag`
   - rerun a `generic_sparse` anchor at the same training length
4. Keep dense LISTA out of the Kuramoto recovery path unless a specific ablation needs it.
5. Only escalate to representation changes if the tuning-only recovery fails.

Deliverables:
- `docs/planning/kuramoto_recovery_plan.md`
- dedicated Kuramoto sweep / collect scripts
- `docs/EXPERIMENTS.md` next-step update

Done criteria:
- the repo has an executable Kuramoto recovery pilot
- the plan distinguishes tuning risk from representation risk
- the acceptance criteria for promotion vs escalation are explicit

## Subagent C: Supervisor Presentation Notes

Objective:
- Produce presentation-ready notes on why these intrinsic-HD systems were chosen and what the current results imply.

Primary inputs:
- `docs/planning/high_dim_benchmarks_plan.md`
- `docs/EXPERIMENTS.md`
- `results/high_dim_benchmarks_plan_seq8_20260305/system_medians_h1000.md`
- `results/high_dim_benchmarks_plan_seq8_20260305/forecasting_summary.md`

Required tasks:
1. Explain the benchmark choice in terms a research supervisor will care about:
   - intrinsic dimensionality
   - attractor diversity
   - relevance to basin-support alignment
2. Summarize the current results with one clear table and a short narrative.
3. Separate positive-control conclusions from blockers.
4. End with the concrete next experiments, not generic future work.

Deliverables:
- a concise supervisor-facing markdown brief

Done criteria:
- the user can turn the brief directly into 3-5 presentation slides
- the story is honest about current blockers
- `competitive_lv`, `kuramoto`, and `hopfield` each have a distinct role in the narrative
