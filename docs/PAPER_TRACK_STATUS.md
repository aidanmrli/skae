# Paper Track Status

Date: April 7, 2026
Paper-critical live queue status last refreshed: `2026-04-07 EDT`

## Goal

The paper target is now explicit:

- keep the fair `200k` forecasting packet, the hard-system packet, and the existing mechanism packet as **decision-grade supporting evidence** rather than the lead live branch
- move the lead live branch to **deterministic transition-rich basin partitioning and classification** on simple toy systems that we can analyze mechanistically in the paper
- execute that branch through the tests-first plan in [docs/planning/transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md)
- use [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md) as the current ground-truth planning note for ablation design choices when iterating on items `3` and `4` of that transition-rich plan, and replace planning assumptions with experiment-backed conclusions once those axes are run systematically
- prefer deterministic native-plot `2D` toy systems first, with follow-on `3D` variants only after the `2D` candidates are calibrated
- require candidate systems to have `3-10` endpoint basins and stable long-rollout endpoint labels
- for the frozen first-pass pair, keep the endpoint-conditioned crossing fractions in the acceptable `0.30-0.70` range
- for the explicit chart-switching transfer family, use source-neighborhood transfer fractions together with inner-core retention instead of the old endpoint-conditioned crossing gate
- distinguish **endpoint basin** from **finite-horizon transition**; the new branch is intentionally transition-rich even though endpoint basins remain well defined
- write **tests before any system-specific code**, then calibrate toy systems before queueing model sweeps
- keep training-time method design label-free: do not assume known basin counts or basin labels outside benchmark diagnostics
- prioritize metrics that explain partition reuse, local predictive structure, and transition failure modes over MSE-only reporting
- keep the matched zero-sparsity MLP as the main causal control whenever we ask whether explicit sparsity itself matters
- treat **`200k` as the only main-text training budget** for the frozen benchmark and hard-system supporting packets
- treat the dense LISTA Stage 1-4 chain as **appendix-only comparator-selection provenance**
- treat the historical MLP `+ block_diagonal K` fairness controls as **invalidated** and replace them with the repaired March 17/19 rerun summaries; full audit: [docs/mlp_block_k_audit_20260317.md](/home/mila/l/lia/skae/docs/mlp_block_k_audit_20260317.md)
- keep the reproduction-grade senior-coauthor handoff packet in [docs/review_main_results_tables_20260314.md](/home/mila/l/lia/skae/docs/review_main_results_tables_20260314.md) and [docs/review_main_results_tables_20260314.tex](/home/mila/l/lia/skae/docs/review_main_results_tables_20260314.tex) as the exact source for the frozen supporting packets

Active execution note:
- The forecasting packet is now decision-grade, and the raw-source seed-statistics companion report is in [docs/PAPER_SEED_STATISTICS_20260331.md](/home/mila/l/lia/skae/docs/PAPER_SEED_STATISTICS_20260331.md). It verifies raw-vs-collector agreement and records the remaining raw finite-value coverage gaps explicitly.
- The benchmark best-periodic packet is still seed-clean at the raw finite-value level. The residual finite-value forecasting gaps are narrower: fixed-cadence late-horizon Hopfield and embedded-multiwell rows, corrected `4`-basin CLV block-diagonal LISTA, and repaired Hopfield `N=64` block-diagonal MLP.
- The current lead blocker is no longer queue completion. The deterministic
  transition-rich Stage 1 screen and both dependent mechanistic passes are now
  collected, so the blocker is now claim selection and benchmark positioning.
- The current ablation-design source of truth for the next interpretability
  loop is
  [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md).
  It should guide matrix-structure, reset, loss, and diagnostic sweeps for
  plan items `3` and `4` until those axes are executed and written back into
  the live docs.
- The forward interpretability scope is now frozen to `17` systems and no
  others:
  - native transition-rich trio:
    `multiwell_strong_transition`, `gated_local_linear`,
    `gated_transfer_linear`
  - Claude-catalog subset:
    `arrested_spiral`, `cal_asymmetric_3`, `cal_high_cross_3`,
    `cal_hexagon_6`, `cal_octagon_8`, `cal_pentagon_5`, `cal_square_4`,
    `checkerboard_potential`, `duffing_triple_well`, `snic_multi`,
    `transition_routes_4`, `var_depth_gradient_4`, `var_diamond_4`,
    `var_l_shape_5`
- The conceptual inventory in
  [docs/planning/transition_rich_system_inventory_20260406.md](/home/mila/l/lia/skae/docs/planning/transition_rich_system_inventory_20260406.md),
  the elite sketches in
  [docs/planning/transition_rich_elite_system_sketches_20260406.md](/home/mila/l/lia/skae/docs/planning/transition_rich_elite_system_sketches_20260406.md),
  and the companion figures under
  [docs/figures/transition_rich_inventory_20260406](/home/mila/l/lia/skae/docs/figures/transition_rich_inventory_20260406)
  now serve as historical design-space provenance only. Do not use their
  broader `16`-system / `8`-system shortlists as the live experiment scope.
- The worktree now also has an artifact-backed audit of the already
  implemented Claude catalog in
  [docs/planning/claude_catalog_audit_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_audit_20260407.md)
  with a companion figure under
  [docs/figures/claude_catalog_audit_20260407](/home/mila/l/lia/skae/docs/figures/claude_catalog_audit_20260407):
  - `112` systems are registered in `skae/claude_catalog`
  - the combined grounded screen now covers `83`
  - `29` implemented systems remain unscreened
  - `12` screened systems are now accepted under the fast-screen gates, with an
    `8`-system strict-crossing core
  - `var_depth_gradient_4` is now part of that strict core and `cal_hexagon_6`
    is now part of the relaxed-accept subset after targeted retune refreshes,
    while `cal_octagon_8` remains a screened high-basin near miss
  - the companion packet now includes a combined audit atlas plus separate
    strict-crossing and accepted-pass portrait galleries
  - the implemented Claude catalog should therefore be treated as a grounded
    small benchmark pool plus retune frontier rather than as an already
    validated large benchmark packet
  - use that audit to describe what is grounded and what remains frontier
  - do not treat the older `6`-system packet in
    [docs/planning/claude_catalog_handoff_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_handoff_20260407.md)
    as the active recommendation; forward Claude-catalog experiments are now
    restricted to the fixed `14`-system subset above
  - the same supersession is now also written in senior-coauthor-facing
    language in
    [docs/planning/claude_catalog_senior_review_packet_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_senior_review_packet_20260407.md),
    so the protocol and active scope are readable without relying on internal
    code names
- The first frozen `2D` pair remains useful, but it is now understood as the
  secondary half of a three-system suite:
  - `gated_local_linear` is now the cleanest oracle-validated chart-switching
    toy
  - `multiwell_strong_transition` is the weaker shared-corridor toy
  - `gated_transfer_linear` is the explicit-transfer stress test where periodic
    decode/re-encode may still help at chart changes, but the oracle state-space
    gap is only modest
- The first implementation tasks are complete locally:
  - write tests
  - add the minimal interface scaffolding
  - calibrate deterministic toy systems
  - freeze two native-plot secondary `2D` candidates
  - implement and calibrate the explicit-transfer toy
- The paper-facing 100-trajectory figure now exists at [transition_rich_100_trajectories.svg](/home/mila/l/lia/skae/docs/figures/transition_rich_20260331/transition_rich_100_trajectories.svg).
- The transfer-system figure packet now also exists under
  [docs/figures/chart_switching_transfer_20260331](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331).
- Short training-entry smokes also pass under [runs/transition_rich_smoke](/home/mila/l/lia/skae/runs/transition_rich_smoke), so the new environments are now verified through the actual training CLI as well as through direct calibration.
- The standardized checkpoint evaluation path now also contains a deterministic flow-branching diagnostic for the frozen `2D` systems, with normalized tolerance sweeps, ground-truth-null thresholds, region-wise breakdowns, and companion `2D` visual artifacts.
- A new oracle state-space benchmark-validity read is now complete under
  [results/transition_rich_oracle_chart_switch_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_chart_switch_20260401).
  Its key consequence is a stricter benchmark ranking:
  - `gated_local_linear` is the clean main-text mechanistic positive
  - `gated_transfer_linear` should be written as the harder explicit-transfer
    stress test, not as the sole chart-switching flagship
  - `multiwell_strong_transition` remains a secondary toy
- A second oracle benchmark-validity read is now also complete under
  [results/transition_rich_oracle_refresh_cadence_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_refresh_cadence_20260401).
  It sharpens the same role split:
  - `gated_local_linear` is the clean periodic-refresh positive
  - `multiwell_strong_transition` is weaker and needs faster refresh
  - `gated_transfer_linear` is strongly cadence-sensitive and should be read as
    a hard transfer stress test whose relevant comparison is stale local charts
- The full seed-robust screening matrix on the calibrated three-system suite is
  now complete:
  - Stage 1 array `9135303` finished with `26/27` completed cells; the single
    failed task was `lista_dense_promoted_stage4 x multiwell_strong_transition x seed_2`
  - collect `9135304` completed `0:0`
  - comparisons `9135305-9135307` completed `0:0`
  - post hoc chart-change attribution `9135358` completed `0:0`
  - post hoc support local-linearity `9135411` completed `0:0`
- The collected read fixes the current paper-facing role split more sharply than
  the oracle-only stage:
  - `gated_local_linear` is the clean learned-model mechanistic positive
  - `gated_transfer_linear` is also a clean forecasting positive, but the
    stronger localization claim on true chart-change windows is negative
  - `multiwell_strong_transition` is not a clean forecasting positive at the
    paper `200k` budget
- The support local-linearity pass is a real positive across the suite, but the
  matched zero-sparsity MLP is also strong, so the current mechanism claim is
  about reusable local partitions rather than about explicit sparsity alone.
- The remaining clearly undercovered older mechanism artifacts are corrected competitive-LV support alignment under [results/zero_sparse_mechanisms_20260321/competitive_lv_representation_followup/support_alignment](/home/mila/l/lia/skae/results/zero_sparse_mechanisms_20260321/competitive_lv_representation_followup/support_alignment) with only seeds `0,1,2`, and the direct Kuramoto mode-support audit with only `5` seeds per root and sampling strategy; neither is the lead paper blocker anymore.
- Do not queue another broad benchmark or hard-system rerun by default. The
  immediate work is to turn the collected transition-rich read into a clean
  paper claim and plan any next runs only on the fixed `17`-system shortlist
  above. Do not reopen broader conceptual-inventory or full Claude-catalog
  selection for this branch; the only remaining selection question is the run
  order within the fixed shortlist and whether the single missing dense-LISTA
  `multiwell` seed is worth finishing.

LISTA follow-up priority note:
- The highest-leverage LISTA follow-up is no longer hypothetical. The compact recurring-support local-linearity study has already been executed on `multiwell_strong_transition` and targeted smaller-`dt` `kuramoto`, and the corrected `competitive_lv` branch is now explicitly out of scope because the repaired clustering rerun stayed negative.
- The first-pass read narrows rather than broadens the LISTA claim: multiwell shows reusable support-conditioned local dynamics for all three roots, but coverage stays below the planned acceptance gate and thresholded `generic_sparse` remains competitive, especially on one-step fits; Kuramoto is a clean negative with no retained recurring support groups on any evaluated seed.
- Corrected CLV has now failed the same filter more decisively than the first summary suggested. It did not clear the support-view clustering gate, it kept negative cosine separation on every clean `4`-basin paper root, and its discovered support-view cluster counts collapse to `5/2/1` across seeds rather than a stable `4`.
- Expanded plan: [docs/planning/lista_narrative_strengthening_followups_20260313.md](/home/mila/l/lia/skae/docs/planning/lista_narrative_strengthening_followups_20260313.md)

Fairness blocker status (updated March 20, 01:00 EDT):
- The historical `generic_sparse + block_diagonal K` MLP fairness controls are **not decision-grade**. A March 17 audit showed that the runs were launched with `K_STRUCTURE=block_diagonal`, but `GenericKM` ignores that setting and still learns a dense latent transition. Full audit: [docs/mlp_block_k_audit_20260317.md](/home/mila/l/lia/skae/docs/mlp_block_k_audit_20260317.md).
- The implementation blocker is cleared: `GenericKM` honors dense / diagonal / block-diagonal `K`, the shared structure tests pass locally, and the repaired March 17/19 fairness-control wave is fully integrated into the paper-facing packet and [docs/review_main_results_tables_20260314.tex](/home/mila/l/lia/skae/docs/review_main_results_tables_20260314.tex).
- The follow-up execution blocker is also cleared: the March 17 wrappers omitted the repo’s separate checkpoint-only `H1500-H3000` reevaluation stage, but that gap is now repaired and March 19 extension chain `8986054 -> 8986055 -> 8986056` completed `0:0`.
- Affected historical artifacts:
  - **Competitive LV:** array `8922796` finished `6/6`, but the reported MLP `block-K` tie (`0.1254` versus `0.1254`) is invalid as a structure-isolation result
  - **Kuramoto:** array `8922810` finished `5/5`, but the reported `generic_sparse_blockdiag=31.80` row is an invalid historical artifact
  - **Hopfield:** array `8922811` finished `15/15`, but the apparent MLP `block-K` match to sparse MLP is likewise invalid
- Result: the paper-critical structure-isolation question is now closed for every cited Kuramoto / Hopfield / corrected-CLV family.
- March 17 fairness reruns are fully refreshed through `H3000`, and the March 19 retry1 Kuramoto mirrors are now also collected and compared.
- Broader paper-wide causal language should now cite those fresh March 17/19 summaries in place of the invalid historical rows, but it no longer needs to treat the fairness thread as an open queue blocker.
- The detailed execution plan is now historical provenance in [docs/archive/planning/generic_sparse_blockdiag_patch_and_paper_rerun_plan_20260317.md](/home/mila/l/lia/skae/docs/archive/planning/generic_sparse_blockdiag_patch_and_paper_rerun_plan_20260317.md).
- Immediate focus is no longer the fairness rerun wave; it is the deterministic transition-rich basin-partitioning branch.
- Broader cleanup note: other historical `GenericKM + structured Koopman` artifacts should also be redone later, but they are not part of the first paper-critical rerun wave.
- Diagonal policy (March 11): retire `lista_diagonal` from active paper scope. Keep the existing diagonal-K results only as historical context; do not spend more queue budget on diagonal reruns or include diagonal in future paper-facing rebuilds.

## Consolidated Paper-Facing Families

For drafting, compress the completed evidence into the family map below rather than citing each queue-era branch as its own experiment, and make the new lead branch explicit.

| Family | Merge these experiments | Main paper question | Writing rule |
|---|---|---|---|
| **Transition-rich basin partitioning** | tests-first toy-system calibration; transition-rich `multiwell` bridge extension; new `gated_local_linear` family; `gated_transfer_linear` chart-switching transfer family; transition diagnostics; deterministic flow-consistency / flow-branching read; recurring-support local-linearity reuse metrics | Can learned supports define reusable, label-light partitions on deterministic systems with common transitions, while preserving deterministic flow consistency? | This is the lead live family. Write it around the collected trained-model read: `gated_local_linear` as the clean mechanistic positive, `gated_transfer_linear` as the hard transfer stress test, and `multiwell_strong_transition` as a weaker secondary toy unless it is rescued by a targeted follow-up. |
| **Cross-system forecasting** | fair `200k` benchmark checkpoint family; matched zero-sparsity MLP benchmark extension; full-benchmark block-diagonal dense-opt transfer | What is the benchmark headline versus the MLP anchor once the dense comparator is fixed, and does explicit sparsity help beyond the same MLP with `lambda_sparse=0`? | Cite the fair `200k` benchmark as the supporting quantitative anchor. Do not let it crowd out the new transition-rich branch. |
| **Hard-system forecasting** | `dt`-rescue audit; focused smaller-`dt` Kuramoto/Hopfield follow-ups; long-horizon reevaluation of those same checkpoints; Kuramoto robustness/dimension sweeps; matched hard-system parity; matched block-diagonal fairness controls; higher-basin Hopfield / CLV robustness probes; matched zero-sparsity MLP hard-system extension | Where do step size, structure, and the sparsity penalty help, and where do LISTA families still fail? | Write this as one connected limitation/support family, not as a separate live execution branch. |
| **Basin-support and mechanism** | broad support-alignment audit; label-free clustering v2; direct Kuramoto mode-support audit; corrected `competitive_lv` representation follow-up; March 13 recurring-support local-linearity; matched zero-sparsity MLP mechanism extension | When do sparse supports align with basins and support local linear reasoning, and do those effects persist without an explicit sparsity penalty? | Use this as prior context and as the metric starting point for the new transition-rich branch. |
| **Appendix-only provenance** | dense LISTA recipe-selection/tuning sweeps; matched `50k` `v4` four-model audit | What tuning/provenance material justifies the fixed comparator choices and historical benchmark context? | Keep appendix-only. Do not present hyperparameter selection as a main result family. |

## Current Best Evidence

Paper-priority note:
- The live open branch is now the deterministic transition-rich basin-partitioning plan in [docs/planning/transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md).
- The evidence below remains the fixed supporting packet while that branch is
  being interpreted and written into the paper story.
- Main-text benchmark claims should be drawn from the `200k` results below.
- The older `50k` `v4` matrix is appendix-only historical context unless a reviewer specifically asks for the matched four-model snapshot.
- For the dense MLP-versus-LISTA benchmark, “fair” means matched systems, pass-2 `dt`, `200k` budget, `target_size=256`, `sequence_length=8`, and matched `[64,64]` front-end hidden widths/depths. It is not a strict equal-parameter comparison, because LISTA also keeps its learned recurrent encoder matrix on top of that shared front end.
- A matched zero-sparsity MLP extension of this same benchmark family is now collected under [results/paper_zero_sparse_benchmark_200k_20260321](/home/mila/l/lia/skae/results/paper_zero_sparse_benchmark_200k_20260321); it uses the same `200k` recipe as `generic_sparse_ns200k_best`, except `lambda_sparse = 0.0`, and the current table-facing forecasting packet now uses exact 10-seed coverage for every displayed benchmark and hard-system row. Its current read is mixed: zero-sparsity broadens good-system coverage on some late-horizon slices, but it is no longer the benchmark-median winner at `H1000`.

### Lead live branch: deterministic transition-rich basin partitioning

- The locked execution plan is [docs/planning/transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md).
- The current design-source companion for the unrun ablation axes is
  [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md).
  Treat it as planning ground truth for the next loop over items `3` and `4`,
  not as already-validated evidence.
- The required test suite now exists and passes, along with the supporting calibration module [skae/transition_calibration.py](/home/mila/l/lia/skae/skae/transition_calibration.py) and the reproducible calibration entry point [tools/calibrate_transition_system.py](/home/mila/l/lia/skae/tools/calibrate_transition_system.py).
- Two native-plot `2D` deterministic candidates are now frozen from ground-truth calibration on the fixed `17x17` grid:
  - `multiwell_strong_transition`
    - `5` endpoint basins
    - per-endpoint crossing fractions `0.500 / 0.500 / 0.500 / 0.500 / 0.585`
    - overall crossing `0.512`
    - label stability `1.000`
  - `gated_local_linear`
    - `3` endpoint basins
    - per-endpoint crossing fractions `0.670 / 0.577 / 0.539`
    - overall crossing `0.599`
    - label stability `1.000`
- The explicit-transfer toy is now implemented and calibrated:
  - `gated_transfer_linear`
    - `3` endpoint basins
    - source-neighborhood transfer fractions `0.333 / 0.444 / 0.333`
    - overall source-neighborhood transfer `0.370`
    - core-retention fractions `1.000 / 1.000 / 1.000`
    - overall core retention `1.000`
    - label stability `1.000`
- The paper-usable mechanics figures for that transfer toy now exist:
  - [gated_transfer_linear_region_map.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_region_map.svg)
  - [gated_transfer_linear_chart_trajectories.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_chart_trajectories.svg)
  - [gated_transfer_linear_endpoint_trajectories.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_endpoint_trajectories.svg)
  - [gated_transfer_linear_transfer_summary.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_transfer_summary.svg)
- The same calibration gives one current caveat:
  - `multiwell_gradient` under the new corridor-aware labeling is not yet a clean low-transition control (`overall crossing = 0.401`, center-basin crossing `0.242`), so it should remain a secondary reference until we decide whether to retune it.
- The branch now also has a paper-usable mechanics figure:
  - [transition_rich_100_trajectories.svg](/home/mila/l/lia/skae/docs/figures/transition_rich_20260331/transition_rich_100_trajectories.svg)
- Updated interpretation of those two frozen systems:
  - they are good secondary toys for transition handling and local chart
    mechanics
- A new oracle chart-switch benchmark-validity read is now complete under
  [results/transition_rich_oracle_chart_switch_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_chart_switch_20260401):
  - `gated_local_linear`
    - oracle-vs-global gain at `H20` on all starts: `0.129974`
    - oracle-vs-fixed-chart gain at `H20` on all starts: `0.0219174`
  - `gated_transfer_linear`
    - oracle-vs-global gain at `H20` on all starts: `0.0206368`
    - oracle-vs-fixed-chart gain at `H20` on all starts: `0.00237432`
    - oracle-vs-global gain on transfer starts: `-0.0557357`
    - oracle-vs-fixed-chart gain on transfer starts: `0.00993001`
  - `multiwell_strong_transition`
    - oracle-vs-global gain at `H20` on all starts: `0.0144757`
    - oracle-vs-fixed-chart gain at `H20` on all starts: `0.00539429`
- Revised benchmark role split:
  - `gated_local_linear` is the cleanest chart-switching positive
  - `gated_transfer_linear` is the explicit-transfer stress test
  - `multiwell_strong_transition` is the weaker shared-corridor toy
- A second oracle benchmark-validity read is now complete under
  [results/transition_rich_oracle_refresh_cadence_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_refresh_cadence_20260401):
  - `gated_local_linear`
    - median dwell: `171`
    - recovered oracle fraction at `H20`: `0.988651` at `c=2`, `0.90577` at
      `c=5`, `0.665843` at `c=10`
    - largest cadence preserving at least `90%` of the oracle-vs-fixed gain on
      all starts: `c=5`
  - `multiwell_strong_transition`
    - median dwell: `92`
    - recovered oracle fraction at `H20`: `0.968223` at `c=2`, `0.858159` at
      `c=5`, `0.639087` at `c=10`
    - largest cadence preserving at least `90%` of the oracle-vs-fixed gain on
      all starts: `c=2`
  - `gated_transfer_linear`
    - median dwell: `33`
    - recovered oracle fraction at `H20`: `0.88425` at `c=2`, `0.503731` at
      `c=5`, `0.0577701` at `c=10`
    - largest cadence preserving at least `90%` of the oracle-vs-fixed gain on
      all starts: only `c=1`
    - on transfer starts, oracle refresh still beats stale local charts but not
      the global fit
- The standardized evaluation stack now has one additional forecast-side diagnostic for these systems:
  - sweep a normalized same-state tolerance on pooled rollout states
  - set the next-step divergence threshold from the ground-truth simulator so the true system is exactly zero under the chosen sweep
  - report close-pair counts, flow-branching rate, next-step / `K`-step divergence severity, and basin-core vs transition-region breakdowns
- First model pass is also fixed:
  - `generic_sparse`
  - matched zero-sparsity MLP
  - promoted dense LISTA
  - `lista_blockdiag` only on the strongest positive case if the screening read is genuinely informative
- The Stage 1 execution wrapper and branch-specific summary collector now exist:
  - [tools/build_transition_rich_screening_tasks.py](/home/mila/l/lia/skae/tools/build_transition_rich_screening_tasks.py)
  - [tools/summarize_transition_rich_screening.py](/home/mila/l/lia/skae/tools/summarize_transition_rich_screening.py)
  - [scripts/collect_transition_rich_screening.sh](/home/mila/l/lia/skae/scripts/collect_transition_rich_screening.sh)
  - [scripts/queue_transition_rich_screening_stage1_20260401.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_screening_stage1_20260401.sh)
- The post hoc chart-change attribution pass now also exists as a separate,
  scientifically cleaner stage:
  - [tools/evaluate_transition_rich_chart_change_attribution.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_chart_change_attribution.py)
  - [tools/collect_transition_rich_chart_change_attribution.py](/home/mila/l/lia/skae/tools/collect_transition_rich_chart_change_attribution.py)
  - [scripts/collect_transition_rich_chart_change_attribution.sh](/home/mila/l/lia/skae/scripts/collect_transition_rich_chart_change_attribution.sh)
  - [scripts/queue_transition_rich_chart_change_attribution_20260401.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_chart_change_attribution_20260401.sh)
- The post hoc support-local-linearity pass now also exists as a separate,
  scientifically cleaner stage:
  - [tools/evaluate_transition_rich_support_local_linearity.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_support_local_linearity.py)
  - [tools/collect_transition_rich_support_local_linearity.py](/home/mila/l/lia/skae/tools/collect_transition_rich_support_local_linearity.py)
  - [scripts/collect_transition_rich_support_local_linearity.sh](/home/mila/l/lia/skae/scripts/collect_transition_rich_support_local_linearity.sh)
  - [scripts/queue_transition_rich_support_local_linearity_20260401.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_support_local_linearity_20260401.sh)
- Undertrained smoke validation exists only as a tooling check, not as a paper
  result, under
  [results/transition_rich_smoke_chart_change_attr_20260401](/home/mila/l/lia/skae/results/transition_rich_smoke_chart_change_attr_20260401):
  it shows a small positive chart-change localization delta but no meaningful
  transfer-trajectory gain, which is exactly the expected “pipeline works,
  science not claimed yet” outcome for a `5`-step smoke root.
- Undertrained support-local-linearity smoke validation also exists only as a
  tooling check, not as a paper result, under
  [results/transition_rich_smoke_support_local_linearity_20260401](/home/mila/l/lia/skae/results/transition_rich_smoke_support_local_linearity_20260401):
  on the collected `generic_sparse` transfer smoke, retained support groups
  cover `0.858` of all trajectories, `0.865` of chart-switch trajectories, and
  `0.700` of transfer trajectories, with `H20` local/global/shuffled latent
  NRMSE `0.0701 / 0.1309 / 0.2018` and weighted endpoint / source-endpoint-pair
  purity `1.000 / 0.891`; that is the right nondegenerate tooling read, but it
  is still not a paper claim because the smoke checkpoints trained for only `5`
  steps.
- The collected Stage 1 screen is now the real model-side read:
  - `26/27` runs completed under
    [results/transition_rich_screening_stage1_20260401](/home/mila/l/lia/skae/results/transition_rich_screening_stage1_20260401)
  - `gated_local_linear` is a clean positive for all three roots, with
    `H1000` best-periodic medians `0.0890151`, `0.000710212`, `0.000783794`
    against no-reencode medians `27651.5`, `2.86748`, `206.989`
  - `gated_transfer_linear` is also a clean positive for all three roots, with
    `H1000` best-periodic medians `1.7954`, `1.80077`, `1.09863` against
    no-reencode medians `6.54401e+20`, `2.45499e+34`, `1.25639e+12`
  - `multiwell_strong_transition` is not a clean forecasting positive at the
    paper budget; `H100` and `H500` best-periodic medians remain unstable and
    enormous across roots
- The collected chart-change attribution pass is now also the real model-side
  causal read on the explicit-transfer toy:
  - summary:
    [results/transition_rich_chart_change_attribution_20260401/summary.md](/home/mila/l/lia/skae/results/transition_rich_chart_change_attribution_20260401/summary.md)
  - all three localization deltas are negative for all three roots, so the
    stronger claim that learned periodic gains localize at true chart-change
    windows is not currently supported
- The collected support local-linearity pass is also now the real
  partition-reuse read:
  - summary:
    [results/transition_rich_support_local_linearity_20260401/summary.md](/home/mila/l/lia/skae/results/transition_rich_support_local_linearity_20260401/summary.md)
  - `gated_local_linear` is the strongest reusable-partitions positive,
    `gated_transfer_linear` is intermediate, and `multiwell_strong_transition`
    is weakest on retained-coverage
  - the matched zero-sparsity MLP is also strong, so this is not a sparse-only
    mechanism win
- Interpretation:
  - the branch now has a valid three-system deterministic suite before any
    model training
  - the suite now has two independent ground-truth benchmark-validity reads:
    one for per-step chart switching and one for periodic refresh cadence
  - the branch now also has the first collected model-side diagnostic read, and
    it supports a narrower claim than the most ambitious version of the story:
    strong periodic-refresh positives on `gated_local_linear` and
    `gated_transfer_linear`, but not a clean `multiwell_strong_transition`
    positive and not a chart-localization win on the transfer toy
- The supporting benchmark, hard-system, and mechanism evidence below remains the context for interpreting this new branch.

### 1. Supporting benchmark packet: one fair `200k` checkpoint family with `H100-H3000` coverage

- The table-facing benchmark family should now be read from [results/paper_zero_sparse_benchmark_200k_20260321/collect/forecasting_rows.csv](/home/mila/l/lia/skae/results/paper_zero_sparse_benchmark_200k_20260321/collect/forecasting_rows.csv), [results/paper_zero_sparse_benchmark_200k_20260321/collect/paper_benchmark_summary.json](/home/mila/l/lia/skae/results/paper_zero_sparse_benchmark_200k_20260321/collect/paper_benchmark_summary.json), and [results/paper_zero_sparse_benchmark_200k_20260321/fixed_cadence_periodic_100/fixed_cadence_summary.json](/home/mila/l/lia/skae/results/paper_zero_sparse_benchmark_200k_20260321/fixed_cadence_periodic_100/fixed_cadence_summary.json), not from the older mixed-coverage benchmark summaries.
- The verified best-periodic cross-system medians are:
  - sparse MLP: `2.947e-4 / 0.0051 / 0.0240 / 0.0795 / 0.1123 / 0.1591 / 0.2201`
  - zero-sparsity MLP: `3.021e-4 / 0.0063 / 0.0353 / 0.1067 / 0.1351 / 0.1738 / 0.1951`
  - promoted dense LISTA: `3.440e-4 / 0.0047 / 0.0250 / 0.0449 / 0.0627 / 0.0880 / 0.1039`
  - horizons are `H100/H500/H1000/H1500/H2000/H2500/H3000`
- The verified good-system counts are:
  - sparse MLP: `27/25/25/26/25/25/25`
  - zero-sparsity MLP: `27/26/26/27/27/27/27`
  - promoted dense LISTA: `28/25/26/25/25/25/25`
- Interpretation:
  - sparse MLP is still the clean anchor at `H100` and `H1000`
  - promoted dense LISTA is the strongest cross-system LISTA result and is median-best at `H500` and `H1500-H3000`
  - zero-sparsity does not produce the benchmark-median `H1000` win implied by the earlier mixed-coverage snapshot, but it does improve late-horizon coverage on more systems than it wins by median
  - the paper therefore needs a genuine three-way benchmark read rather than a sparse-versus-dense summary plus a side-note control
- Under one global deployment-like cadence `periodic_100`, the verified late-horizon medians are:
  - sparse MLP: `0.1070 / 0.2140 / 0.3690 / 0.6148`
  - zero-sparsity MLP: `0.1729 / 0.4542 / 0.7041 / 0.9349`
  - promoted dense LISTA: `0.0744 / 0.2688 / 0.4437 / 0.6508`
  - horizons are `H1500/H2000/H2500/H3000`
- Fixed-cadence interpretation:
  - explicit sparsity still helps materially relative to the matched zero-sparsity MLP under one deployment-like cadence
  - promoted dense LISTA remains the late-horizon fixed-cadence median winner in the benchmark packet, so the deployment-style story is again three-way rather than a sparse-only positive
- The dense-optimizer block-diagonal transfer still does not rescue block-diagonal LISTA globally; keep block-diagonal claims restricted to targeted hard-system evidence and explicit fairness-control tables.

### 2. Appendix-only dense LISTA comparator-selection provenance

This is supporting provenance for the dense comparator used in Section 1. It should be mentioned only briefly in the main text and detailed in the appendix if needed.

- The dense-LISTA easy-system parity Stage 1 is complete under [dense-LISTA easy-system Stage 1 summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/collect/paper_benchmark_summary.md):
  - same dense LISTA architecture and benchmark-selected `dt` on all `8` target systems
  - Stage 1 changed only `num_steps`, `lr`, and `k_matrix_lr`
- Best Stage-1 recipes against the fixed `generic_sparse` anchor:
  - `lista_dense_ns100k_lr5em5_klr5em6_wd1em4` wins `6/8` target systems with median dense/generic ratio `0.8699`
  - `lista_dense_ns200k_lr5em5_klr5em6_wd1em4` wins `5/8` target systems with median ratio `0.7888`
- Best per-system dense recipe still loses on:
  - `competitive_lv` (`1.764x` vs `generic_sparse`)
  - `duffing` (`1.041x` vs `generic_sparse`)
- Positive details:
  - all `9` Stage-1 dense-LISTA recipes keep `8/8` target systems under the good-forecast band
  - no Stage-1 dense recipe is catastrophic on the target set
- Primary audit files:
  - [Stage-1 paper summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/collect/paper_benchmark_summary.md)
  - [Stage-1 forecasting summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/collect/forecasting_summary.md)
  - [best win-count comparison](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/compare/lista_dense_ns100k_lr5em5_klr5em6_wd1em4_vs_generic_sparse/forecasting_comparison.md)
  - [best median-ratio comparison](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/compare/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_vs_generic_sparse/forecasting_comparison.md)
- Interpretation:
  - The dense-LISTA gap on the easier accepted-default systems is now clearly partly optimization-limited, not purely architectural.
  - This is strong support for a fairness-preserving dense-LISTA recovery story.
  - It is still not enough to claim dense LISTA is better than `generic_sparse` on most systems overall, because the result is limited to the targeted `8`-system subset and the holdouts remain real.
- The coefficient-only Stage 2 holdout sweep is now complete under [Stage-2 forecasting summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage2_20260308/collect/forecasting_summary.md):
  - `duffing` is flipped only by the specialized `100k, sc=0.012` recipe (`0.0182` vs `0.0309`, `0.590x`)
  - `competitive_lv` is not flipped by any coefficient-only recipe
  - the best global-compromise holdout recipe is `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3`
- The exact `8`-system validation Stage 3 is now complete under [Stage-3 paper summary](/home/mila/l/lia/skae/results/dense_lista_recipe_validation_stage3_20260309/collect/paper_benchmark_summary.md):
  - `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3` wins `6/8` shared systems vs `generic_sparse`
  - shared-system median `H1000` best-periodic ratio is `0.6928`
  - `8/8` systems stay under the good-forecast band with `0` catastrophic systems
  - all seeds are good on all `8` systems
  - the cheaper `100k, sc=0.003` recipe reaches `3/8` wins
  - the Duffing-fixing `100k, sc=0.012` recipe falls to `2/8` wins overall
- Interpretation:
  - the dense-LISTA fairness question is now resolved: promote `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3` as the single fair dense-LISTA recipe
  - the paper story is no longer “LISTA almost catches up if tuned enough”; it is “a fixed dense LISTA architecture recovers most easy-system near-misses with one fair external recipe, but still leaves a persistent `competitive_lv`-style holdout and does not overturn the global `generic_sparse` ranking”
- The promoted dense-LISTA full `29`-system rerun is now complete under [Stage-4 paper summary](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/collect/paper_benchmark_summary.md):
  - same dense LISTA architecture, promoted Stage-3 recipe, and benchmark-selected pass-2 `dt` table
  - compared against the fixed `generic_sparse` `v4` anchor, dense LISTA wins `21/29` shared systems with median dense/generic ratio `0.6455`
  - cross-system median `H1000` best-periodic improves from `0.0328` to `0.0232`
  - good-system count improves from `25/29` to `26/29`
  - there are `0` systems where the promoted dense recipe fails the good-forecast band while `generic_sparse` passes
  - the remaining dense failures are still concentrated on the hard systems, especially `kuramoto` (`48.50`), `hopfield` (`1.578e+06`), and `multiwell_strong_transition_hd` (`4.533e+04`)
- Primary audit files:
  - [Stage-4 paper summary](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/collect/paper_benchmark_summary.md)
  - [Stage-4 forecasting summary](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/collect/forecasting_summary.md)
  - [Stage-4 dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/compare/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3_vs_generic_sparse/forecasting_comparison.md)
- Interpretation:
  - the dense-LISTA parity story is now stronger than the Stage-3 subset result alone: one fixed dense recipe beats the fixed `generic_sparse` anchor on most benchmark systems overall
  - for paper writing, the fair `200k` comparison in Section 1 is the dense-vs-MLP headline; the older Stage-4-vs-`v4` comparison is supporting evidence for recipe promotion, not the main-text benchmark claim
  - when writing that headline, say explicitly that the dense-vs-MLP comparison is width/depth-aligned but not parameter-matched
  - the remaining dense-LISTA paper risk is no longer “can it catch up on the easier systems?”; it is whether the paper cleanly separates the cross-system parity win from the unresolved hard-system failures

### 3. `dt` rescue and hard-system follow-up are now decision-grade: step size is a real bottleneck, Kuramoto has a targeted `200k` rescue, and Hopfield remains MLP-better

- A matched zero-sparsity MLP extension of every paper-facing hard-system setting cited in this section is now collected under [results/zero_sparse_hard_systems_20260321](/home/mila/l/lia/skae/results/zero_sparse_hard_systems_20260321). It covers Kuramoto (`N=16` identical, `N=16` uniform-spread, and the dimension sweep), Hopfield (`N=16` and quarter-step `N=64`), corrected 4-basin competitive LV, and fixed 8-basin competitive LV at both cited step sizes. The recovery chain cleared on March 22 when requeued task `9007966_4` completed and refreshed collectors/comparisons `9016645-9016660` all finished `0:0`.

- The repaired `dt` resolution completed through pass `2`:
  - `15/29` systems accept default `dt`
  - `4/29` systems accept after at least one halving
  - `10/29` systems remain `integration_hard`
- The most important remaining `integration_hard` systems are:
  - `kuramoto` (`selected dt = 0.0125`)
  - `hopfield` (`selected dt = 0.0125`)
  - `lotka_volterra`
  - `multiwell_strong_transition`
  - `multiwell_gradient_hd`
  - `multiwell_rotational_hd`
  - `multiwell_strong_transition_hd`
  - `dysts:DequanLi`
  - `dysts:WangSun`
  - `dysts:LorenzCoupled`
- Current high-dimensional bottlenecks at the selected smaller `dt` are still bad:
  - `kuramoto`, `generic_sparse`, system-median `H1000` best-periodic: `65.7014`
  - `hopfield`, `generic_sparse`, system-median `H1000` best-periodic: `199.4978`
  - `kuramoto`, `lista_blockdiag`, system-median `H1000` best-periodic: `14.2618`
  - `hopfield`, dense LISTA, system-median `H1000` best-periodic: `7.241e+09`
- Primary audit file:
  - [v4 pass-2 `dt` resolution summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/dt_resolution/pass2/dt_resolution.md)
- Interpretation:
  - the benchmark is no longer blocked by queue completion
  - step size is a real scientific bottleneck, not just a scheduling artifact
  - the open paper problem shifted from "finish the rerun" to "what to do with systems that stay hard even after the allowed `dt` rescue"

- The repaired focused intrinsic-HD rerun is complete under [intrinsic-HD `dt` rescue rerun summary](/home/mila/l/lia/skae/results/intrinsic_hd_dt_rescue_20260308_rerun1/forecasting_summary.md):
  - all `48` rows are collected
  - official selection is still based on `evaluation_results_best.json`
- Best current intrinsic-HD arms at `H1000` best-periodic:
  - `kuramoto`:
    - `lista_blockdiag`, `dt=0.0125`, `sp=0.0005`: `14.36`
    - matched `generic_sparse`, `dt=0.0125`, `sp=0.0005`: `25.93`
  - `hopfield`:
    - `generic_sparse`, `dt=0.0125`, `sp=0.0005`: `71.02`
    - best `lista_blockdiag`, `dt=0.0125`, `sp=0.0010`: `80.54`
- Smaller `dt=0.0125` beats `dt=0.025` for both systems in both model families, so the step-size hypothesis is now directly supported by a full focused rerun.
- A diagnostic recollection from `evaluation_results_last.json` shows checkpoint-selection mismatch on Kuramoto:
  - `lista_blockdiag`, `dt=0.0125`, system-median `H1000` improves from `23.40` to `14.64` across the focused pilot grid
  - on the winning `lista_blockdiag`, `dt=0.0125`, `sp=0.0005` arm, the last-checkpoint median is `13.91`
  - this is a diagnostic, not yet the official paper metric, but it shows late training still matters on Kuramoto
- Completed focused Kuramoto `dt=0.00625`, `200k` follow-up under the official short-horizon compare [Kuramoto comparison](/home/mila/l/lia/skae/results/kuramoto_dt00625_200k_compare_20260308/compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md) plus the later long-horizon collector [Kuramoto forecasting summary](/home/mila/l/lia/skae/results/kuramoto_dt00625_200k_compare_20260308/collect/forecasting_summary.md):
  - `generic_sparse`: seed-median `H1000` best-periodic `27.02`
  - dense LISTA: seed-median `H1000` best-periodic `13.84`
  - `lista_blockdiag`: seed-median `H1000` best-periodic `6.98`
  - all five `lista_blockdiag` seeds are good and tightly clustered in `6.89-7.13`
  - the underlying focused summary is again not `H1000`-only: at `H100`, `generic_sparse=0.0343`, `lista_blockdiag=0.1536`, `lista_dense=0.2194`; at `H500`, `generic_sparse=1.4972`, `lista_blockdiag=2.6834`, `lista_dense=4.1222`; the `lista_blockdiag` win appears only at `H1000`
  - the same run family now also has explicit `H1500/H2000/H2500/H3000` medians from the collector: `generic_sparse = 547.37 / 1.208e+04 / 3.370e+05 / 9.207e+06`, `lista_blockdiag = 10.93 / 14.52 / 17.94 / 21.58`, dense LISTA `= 54.85 / 205.26 / 541.19 / 1519.09`
  - **Fixed-cadence ablation (completed):** `periodic_100` exactly reproduces the official `best_periodic` `H1000` ranking for all three roots (`6.98`, `13.84`, `27.02`), so the long-horizon Kuramoto block-diagonal win is already a fixed-cadence result, not a `best_periodic` oracle artifact. At `H500`, `generic_sparse` still has lower error than `lista_blockdiag` under the fixed cadence.
  - **Checkpoint-selection ablation (completed):** switching from `evaluation_results_best.json` to `evaluation_results_last.json` on the current `dt=0.00625`, `200k` comparison does not change the model ranking or good-band membership (`lista_blockdiag`: `6.98 -> 7.00`, `lista_dense`: `13.84 -> 17.63`, `generic_sparse`: `27.02 -> 29.43`). The older `dt=0.0125` pilot mismatch (`23.40 -> 14.64`) is real but tied to the earlier, superseded setting. Keep `evaluation_results_best.json` as the official paper rule.
- Completed Hopfield `dt=0.00625`, `200k` follow-up under [Hopfield follow-up summary](/home/mila/l/lia/skae/results/hopfield_dt00625_200k_compare_20260309/forecasting_summary.md):
  - `generic_sparse`: seed-median `H1000` best-periodic `3.36`
  - `lista_blockdiag`: seed-median `H1000` best-periodic `8.82`
  - the same file shows the MLP lead at every collected horizon: `H100` `0.0500` vs `0.1075`, `H500` `0.8835` vs `3.5303`, `H1000` `3.3642` vs `8.8212`
  - the long-horizon continuation stays negative as well: `generic_sparse` is `6.61 / 9.17 / 10.96 / 12.23` at `H1500/H2000/H2500/H3000`, versus `12.23 / 12.83 / 13.23 / 13.58` for `lista_blockdiag`
  - both are inside the good-forecast band on the system median, but every-step errors are still enormous for both
- The completed `H3000` reevaluation sharpens the hard-system long-horizon read across the whole Kuramoto/Hopfield family:
  - Kuramoto `N=16`, `dt=0.00625`, `200k`: `lista_blockdiag` remains much better than `generic_sparse` across `H1500/H2000/H2500/H3000` (`10.93 / 14.52 / 17.94 / 21.58` vs `547.37 / 1.208e+04 / 3.370e+05 / 9.207e+06`), but it is out of band beyond `H1000`
  - Kuramoto `N=32`, `dt=0.00625`, `200k`: `lista_blockdiag` stays better across `H1500/H2000/H2500/H3000` (`10.89 / 16.14 / 21.69 / 27.95` vs `13.60 / 24.27 / 42.19 / 75.25` for `generic_sparse`), but it is also out of band at those later horizons
  - Kuramoto uniform-spread `N=16`: the repaired `generic_sparse_blockdiag` mirror is now slightly better than `lista_blockdiag` at every reported horizon (`8.13 / 24.76 / 91.23 / 399.74 / 1724.45` versus `9.53 / 28.41 / 117.46 / 523.72 / 2129.37`), but both repaired structured roots are still catastrophic by `H3000`
  - Hopfield `N=16`, `dt=0.00625`, `200k`: `generic_sparse` stays better at `H1500/H2000/H2500/H3000` (`6.61 / 9.17 / 10.96 / 12.23`) than `lista_blockdiag` (`12.23 / 12.83 / 13.23 / 13.58`); only `generic_sparse` stays in-band through `H2000`
  - higher-basin Hopfield `N=64`, `dt=0.0015625`, `200k`: quarter-`dt` improves errors but remains strongly negative for LISTA, with `generic_sparse = 309.92 / 520.71 / 711.04 / 873.50` at `H1500/H2000/H2500/H3000` versus dense `591.50 / 842.59 / 1046.18 / 1206.93` and targeted blockdiag `735.26 / 1035.51 / 1274.93 / 1461.51`
- Completed Kuramoto `N=32`, `dt=0.00625`, `200k`, `3`-seed confirmation under the short-horizon compare [Kuramoto `N=32` summary](/home/mila/l/lia/skae/results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md) and long-horizon collector [Kuramoto `N=32` forecasting summary](/home/mila/l/lia/skae/results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/collect/forecasting_summary.md):
  - `generic_sparse`: seed-median `H1000` best-periodic `6.65` (all seeds good)
  - `lista_blockdiag`: seed-median `H1000` best-periodic `6.00` (all seeds good, std `0.33`)
- Completed Kuramoto dimension sweep under [Kuramoto dimension summary](/home/mila/l/lia/skae/results/kuramoto_dimension_sweep_dt00625_200k_20260309/collect/kuramoto_dimension_summary.md):
  - dimensions: `N={8,16,24,32,64}`
  - models: `generic_sparse`, promoted dense LISTA, `lista_blockdiag`
  - fixed setting: `dt=0.00625`, `200k`, `5` seeds
  - `H1000` seed-median best-periodic by dimension:
    - `generic_sparse`: `813.57`, `30.18`, `6.71`, `6.68`, `208.93`
    - promoted dense LISTA: `495.07`, `13.44`, `14.99`, `92.28`, `208.71`
    - `lista_blockdiag`: `8.11`, `7.07`, `6.57`, `5.92`, `23.27`
  - seed robustness:
    - `lista_blockdiag` is all-seeds-good at `N=16/24/32`
    - `lista_blockdiag` is median-good but not fully robust at `N=8` (`4/5` good seeds, worst seed `10.89`)
    - `lista_blockdiag` falls out of band at `N=64` (`2/5` good seeds, worst seed `209.20`)
- Completed repaired block-diagonal MLP mirrors for the missing Kuramoto families under the retry1 roots:
  - uniform-spread `N=16`: `generic_sparse_blockdiag` reaches `H1000 = 8.13` with all `5/5` seeds good, beating both `generic_sparse` (`44.46`) and `lista_blockdiag` (`9.53`)
  - dimension sweep `H1000`: `generic_sparse_blockdiag = 10.61 / 6.51 / 5.79 / 5.16 / 208.54` at `N=8/16/24/32/64`
  - on that sweep, the repaired MLP block-`K` mirror beats `generic_sparse` at every `N`, beats `lista_blockdiag` at `N=16/24/32`, but not at `N=8` or `N=64`
  - the suspicious `N=64` row is now audited and decision-grade: it is a real seed-fragility limit (`2/5` good seeds, `3/5` collapsed seeds), not a leftover March 17 invalid-control artifact
- Interpretation:
  - smaller `dt` is the dominant hard-system lever in the current regime
  - `lista_blockdiag` is the strongest hard-system LISTA result on Kuramoto and the only model that cleanly wins the `N=16` three-way comparison there at `H1000`; the shorter-horizon audit shows this is a specifically long-horizon advantage rather than across-horizon dominance
  - the completed repaired fairness mirrors narrow the causal read: block structure alone helps on the targeted `N=16` family, on uniform-spread `N=16`, and across the sweep at `N=16/24/32`, but it does not rescue `N=8` and the `N=64` failure is real but seed-sensitive rather than a provenance glitch
  - promoted dense LISTA does not transfer as a robust Kuramoto solution under this sweep
  - Hopfield is no longer a catastrophic boundary case in the targeted `N=16`, `dt=0.00625`, `200k` setting, but it is still not a structured-LISTA success story because `generic_sparse` remains better through `H2000` and both models are out of band by `H2500-H3000`
  - the remaining scientific limitation is autonomous rollout stability, not whether periodic reencoding can rescue the hard systems at all
  - the remaining Kuramoto paper question is no longer missing fairness controls; it is how to present a targeted moderate-dimension success with an explicit `N=8` and `N=64` limit
- Completed Kuramoto robustness evaluation (uniform frequency spread, `N=16`, `dt=0.00625`, `200k`, `5` seeds) under the short-horizon compare [Kuramoto robustness comparison](/home/mila/l/lia/skae/results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/compare/lista_blockdiag_uniform_spread_vs_generic_sparse_uniform_spread/forecasting_comparison.md) and long-horizon collector [Kuramoto robustness forecasting summary](/home/mila/l/lia/skae/results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/collect/forecasting_summary.md):
  - `lista_blockdiag`: seed-median `H1000` best-periodic `9.53` (4/5 seeds good, std `0.64`)
  - `generic_sparse`: seed-median `H1000` best-periodic `44.46` (0/5 seeds good)
  - `4.7x` improvement; the Kuramoto block-diagonal positive is not a single-regime artifact
  - every-step errors remain catastrophic for both models under heterogeneity; periodic re-encoding is essential

### 4. Appendix-only `50k` `v4` audit: useful for matched four-model symmetry, not for main-text claims

- Completed `v4` full-matrix result (`29` systems, `4` baselines, `3` seeds) under the repaired `dt`-rescue chain:
  - `generic_sparse` is best by cross-system median `H1000` best-periodic (`0.0328`)
  - dense LISTA is second (`0.0388`)
  - block-diagonal LISTA is third (`0.1508`)
  - diagonal-K LISTA is worst (`1.2110`)
- `H1000` good-system counts (`best-periodic < 10`):
  - `generic_sparse`: `25/29`
  - dense LISTA: `24/29`
  - block-diagonal LISTA: `24/29`
  - diagonal-K LISTA: `24/29`
- Shared-system comparison against `generic_sparse`:
  - dense LISTA wins `15/29`
  - block-diagonal LISTA wins `3/29`
  - diagonal-K LISTA wins `3/29`
- Interpretation note:
  - `lista_diagonal` is now retired from active paper scope. Keep the completed diagonal numbers only as historical context; do not allocate new experiment budget to diagonal reruns.
- Primary audit files:
  - [v4 paper summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/paper_benchmark_summary.md)
  - [v4 final forecasting summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/forecasting_summary.md)
  - [dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_dense_vs_generic_sparse/forecasting_comparison.md)
  - [block-diagonal vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md)
- Interpretation:
  - `v4` is still a useful matched four-model snapshot and the cleanest source for the `dt`-resolution audit.
  - It should be treated as appendix-only historical context once the `200k` reruns materially improve the same story.
  - Do not use `v4` as the source of main-text rankings or headline model comparisons when a `200k` counterpart exists.

## Recent Queue Activity

The March 9-20 paper-strengthening program is fully closed and now belongs to the archived planning set. The only live successor branch is the deterministic transition-rich basin-partitioning plan documented at the top of this file.

### Just Closed

- **Kuramoto uniform-spread dense-LISTA completion run:** the March 20 appendix-table audit found that the original March 9 uniform-spread robustness manifest never launched dense LISTA, so the missing row was genuine rather than a collector omission. Completion chain `8989946 -> 8989947 -> 8989948/8989949/8989950` finished cleanly under [results/kuramoto_uniform_spread_dense_20260320](/home/mila/l/lia/skae/results/kuramoto_uniform_spread_dense_20260320). Training tasks `8989946_[0-4]` ended between `2026-03-20 02:58 EDT` and `2026-03-20 03:31 EDT`; collector `8989947` finished at `03:32 EDT`, and compare jobs `8989948/8989949/8989950` finished at `03:33 EDT`. The completed row is `0.2081 / 4.1470 / 16.55 / 46.38 / 132.35 / 415.24 / 1530.83` at `H100/H500/H1000/H1500/H2000/H2500/H3000`.
- **Kuramoto uniform-spread valid MLP `+ block-K` rerun:** retry chain `8988801 -> 8988802 -> 8988803/8988804` closed successfully under [results/kuramoto_uniform_spread_gs_blockdiag_retry1_20260319](/home/mila/l/lia/skae/results/kuramoto_uniform_spread_gs_blockdiag_retry1_20260319). Training tasks ended between `2026-03-19 22:33 EDT` and `2026-03-19 23:47 EDT`; the collector finished at `23:48 EDT`, and both compare jobs finished at `23:49 EDT`.
- **Kuramoto dimension-sweep valid MLP `+ block-K` rerun:** retry chain `8988805 -> 8988806 -> 8988807/8988808` also closed successfully under [results/kuramoto_dimension_sweep_gs_blockdiag_retry1_20260319](/home/mila/l/lia/skae/results/kuramoto_dimension_sweep_gs_blockdiag_retry1_20260319). Training tasks ended between `2026-03-19 22:06 EDT` and `2026-03-19 23:54 EDT`; the collector finished at `23:54 EDT`, and both compare jobs finished at `23:55 EDT`.
- **Queue hygiene note:** the first submission attempt (`8988754`, `8988758`) was invalidated by external startup termination after accidental duplicate-submission cleanup. Use only the retry1 roots above for paper-facing interpretation.

### Last 48 hours (completed state)

- **Paper-facing horizon reevaluation:** closed successfully after one tooling repair. Original kickoff array `8929393` failed `300/300` tasks because `run_manifest.tsv` used CRLF line endings. The repaired chain `8933469 -> 8933470 -> 8935507 -> 8936342 -> 8937045 -> 8937956 -> 8938574` completed all `1528` reevaluations and refreshed the benchmark / compare artifacts through `H3000`.
- **Hard-system parity sweep:** closed successfully after a small anchor repair. Stage-2 confirm array `8931671` ended with two failed anchor rows (`87`, `137`); the repair / recollect / resummary chain `8933794 -> 8933795 -> 8933796` completed and produced the final negative-for-LISTA parity summary now used in this document.
- **Corrected `competitive_lv` representation follow-up:** closed after one repair. Wrapper `8933876` completed, support-alignment job `8939086` completed, the first label-free clustering array `8939087` failed `39/39` because checkpoint paths in `lfc_task_specs.tsv` retained a trailing carriage return, and March 13 repair chain `8945129 -> 8945133 -> 8945135` then completed cleanly. The final scientific read is still negative on the clean `4`-basin paper roots.
- **Hopfield quarter-`dt` higher-basin rerun:** fully complete. Base sweep `8926091` and matched fairness control `8926089` both finished; the smaller `dt` improves errors materially but does not move any root into the good band or change the ordering (`generic_sparse` remains best).
- **Other completed paper-side jobs:** the historical matched block-diagonal fairness-control jobs (`8922796`, `8922810`, `8922811`) all finished at the queue level, but their MLP `+ block-K` outputs were later invalidated by the March 17 audit; the CLV high-basin scalability / smaller-`dt` sweeps (`8923108`, `8926090`) and the clean 4-basin CLV retrain / recovery / extension chain (`8922033`, `8922374`, `8922879`) are complete and already reflected in the current paper claims above.
- **Accounting note:** the only other failed job in the last-48-hour accounting window was one unrelated `interactive` shell session (`8936456`); it does not affect paper state.

### Recently completed

- **Paper-facing `H3000` horizon reevaluation:** complete. The refreshed fair benchmark is now split by horizon: `generic_sparse_ns200k_best` is best by cross-system median at `H100/H500/H1000` (`3.146e-4`, `0.0050`, `0.0233`), while promoted dense Stage 4 becomes median-best at `H2000/H2500/H3000` (`0.0627`, `0.0910`, `0.0940`) but falls to `24/29` good systems at `H3000` versus `26/29` for the MLP anchor.
- **Long-horizon fixed-cadence benchmark rescore:** complete offline from the refreshed fair `200k` benchmark JSONs; no queue submission was needed. Under one global `periodic_100`, dense remains breadth-competitive at `H1500-H3000` but no longer has a clean late-horizon median edge: the MLP is better by cross-system median at `H2000/H2500`, and at `H3000` the MLP also keeps better coverage (`20/29` vs `17/29`).
- **Hard-system parity sweep:** complete. `generic_sparse` is best on all `14` confirmed settings; dense LISTA records `0` wins / `10` losses / `4` worse-seed-robustness rows, and `lista_blockdiag` records `0` wins / `8` losses / `6` worse-seed-robustness rows.
- **Corrected `competitive_lv` representation follow-up:** complete. The repaired March 13 clustering rerun wrote the missing `label_free_clustering_v2/` outputs, but the corrected CLV result remains negative for the intended support-based narrative: no clean `4`-basin paper root clears the support-view gate, all keep negative cosine separation, and support-view clustering collapses to `5/2/1` discovered groups across seeds.
- **Competitive LV high-basin smaller-`dt` halving rerun:** complete on array `8926090` (`96/96`). Halving to `dt=0.0025` removes the remaining fixed-system `200k` `lista_blockdiag` seed failures on the higher-basin CLV probes, but `generic_sparse` remains best and `generic_sparse_blockdiag` stays neutral.
- **Kuramoto unique mode-support audit (completed March 10):** All `30/30` array tasks and collector finished under `results/kuramoto_mode_support_audit_20260310/`. The strong negative claim is confirmed: Kuramoto winding-number basins do not have meaningful basin-specific support patterns. Mode-support uniqueness is trivially degenerate — every trajectory has its own unique support (`traj_unique=1.0`), mode supports are singletons, basin consistency is negligible (`0.0625` balanced / `0.309` random), and Hamming geometry is flat (`ratio≈1.0`). This holds identically across all 3 model families (`generic_sparse`, `lista_dense`, `lista_blockdiag`), all 5 seeds, both sampling protocols (`random`, `balanced`), all support modes (`mean`, `majority`, `modal`), and all threshold values (`1e-4` to `1e-1`). This closes the gap left by label-free clustering v2 and directly confirms both claims: basins are not recoverable from latent features, and basins do not have literal reusable mode supports.
- **Broad support-alignment audit on labelable `v4` systems:** complete under [support-alignment summary](/home/mila/l/lia/skae/results/paper_benchmark_support_alignment_20260311_v4_labelable/summary.md). Across `11` valid labelable systems (`132` checkpoints), binary `mode_uniqueness_rate` saturated at `1.0` on all `44/44` system-root medians, while cosine separation still cleanly split the systems: multiwell positive, Duffing negative, Kuramoto negative, and Hopfield mixed.
- **Fair `200k` follow-up benchmark:** complete. `generic_sparse_ns200k_best` is the best full-benchmark root by cross-system median `H1000` best-periodic (`0.0233`), while promoted dense Stage 4 still wins `18/29` shared systems and keeps `26/29` good systems at that horizon.
- **Dense LISTA promoted Stage 4 rerun:** complete. One fixed fair dense recipe wins `21/29` systems against the fixed `generic_sparse` anchor and improves the dense median to `0.0232`.
- **Kuramoto dimension sweep:** complete. `lista_blockdiag` is robust through `N=32`, not fully robust at `N=8`, and no longer in-band at `N=64`.
- **Focused Kuramoto and Hopfield smaller-`dt` follow-ups:** complete. `lista_blockdiag` wins the Kuramoto `dt=0.00625`, `200k` comparison at `H1000`, but the refreshed `H1500-H3000` read reduces that win to a relative-only advantage; Hopfield remains `generic_sparse`-best through `H2000` and both models fail by `H2500-H3000`.
- **Label-free clustering v2:** complete on array `8919951` with collector `8919952`. Multiwell systems are strongly positive, Duffing is weakly positive, and Kuramoto is negative.

### 6. Broad support-alignment audit: binary mode uniqueness saturates, cosine separation carries the real signal

Together with the corrected `competitive_lv` representation rerun and the March 13 local-linearity study, this section belongs to the basin-support/mechanism family rather than a standalone side analysis.

- A matched zero-sparsity MLP extension of this mechanism family is now complete under [results/zero_sparse_mechanisms_20260321](/home/mila/l/lia/skae/results/zero_sparse_mechanisms_20260321). The original wrapper `9007983` was canceled during March 22 recovery, but the refreshed wrapper `9016661` finished the same support-alignment, label-free clustering, Kuramoto support-audit, and recurring-support local-linearity analyses on the no-sparsity MLP checkpoints. The completed Kuramoto support-audit rerun shows that removing the sparsity penalty does not recover reusable mode supports.

- **Result (COMPLETE, March 11, local audit):** Evaluated the canonical `v4` checkpoints on all currently valid labelable benchmark systems under [support-alignment summary](/home/mila/l/lia/skae/results/paper_benchmark_support_alignment_20260311_v4_labelable/summary.md).
  - scope: `11` systems (`duffing`, `8` `multiwell*` variants, `kuramoto`, `hopfield`) x `4` roots x `3` seeds = `132` checkpoints
  - excluded: `competitive_lv`, because the canonical `v4` checkpoints used the invalidated 1-basin configuration
  - settings: `100` trajectories, length `500`, `5000`-step basin rollout, `support_threshold=1e-3`, `support_mode=mean`
- **Concrete result:**
  - binary mode-support uniqueness saturates completely: **all `44/44` system-root medians have `mode_uniqueness_rate=1.0`**
  - support reuse is weak almost everywhere: **`40/44` system-root medians have `mean_basin_consistency < 0.2`**
  - trajectory-level supports are often unique: **`24/44` system-root medians have `trajectory_unique_support_rate = 1.0`**
  - all multiwell system-root medians are positive by cosine separation (`0.250` to `0.706`)
  - Duffing is negative across all roots (`-0.129` to `-0.084`) despite perfect mode uniqueness
  - Kuramoto is negative across all roots (`-0.307` to `-0.264`) despite perfect mode uniqueness; the random `100`-trajectory audit again produces singleton `q=±2` basins, so the apparent `mean_basin_consistency≈0.424` is inflated and not evidence of reusable basin supports
  - Hopfield is mixed: cosine separation is positive across all roots (`0.459` to `0.607`), but `mean_basin_consistency` is only `0.043` for every root and `trajectory_unique_support_rate=1.0` throughout
- **Interpretation:**
  - the literal binary question "does each basin have a unique mode support?" is too weak as a broad benchmark diagnostic, because it returns a perfect score even on known negatives like Duffing and Kuramoto
  - cosine separation reproduces the known qualitative split and should remain the primary support-alignment metric
  - Hopfield currently shows continuous basin separation without reusable sparse support signatures: basin centroids separate, but trajectories do not reuse a stable support within each basin
- **Paper implication:** do not make a benchmark-wide paper claim based on `mode_uniqueness_rate`. Keep the support story tied to multiwell cosine/clustering positives, scope Duffing and Kuramoto as negatives, and treat Hopfield as a mixed continuous-separation-only case.

### 7. Label-free basin recovery: v2 validates label-free clustering on potential-well systems

This is the label-free clustering subfamily inside the same basin-support/mechanism family.

- **v1 result (methodology limitation, March 10):** The initial label-free clustering evaluation used trajectory-mean cosine k-means on 128 trajectories in 256 dimensions. Results:
  - Duffing (2 basins): ARI=`0.134` — all three models produce **identical** scores, confirming the feature extraction protocol (not the encoder) is the bottleneck
  - Kuramoto (5 basins): ARI≈`0` for all models
  - Competitive LV: only 1 basin observed (trivial, now fixed — see competitive_lv retrain below)
- **Root cause:** v1 protocol destroyed per-timestep support signal via trajectory averaging, suffered from concentration of measure (no PCA), and tested only the cosine feature view. The identical Duffing scores across all encoder families confirmed this was a feature-extraction issue.
- **v2 result (COMPLETE, March 10, array `8919951`, collector `8919952`):** Revised evaluation with 6 feature views, PCA to 20d, 256 trajectories, 10 systems.
  - **Multiwell systems (8 variants, 5 basins each): strong positive.**
    - `multiwell_gradient/gradient_hd`: mean ARI `0.976/0.991`, near-perfect recovery (max `1.000`)
    - `multiwell_rotational/rotational_hd`: mean ARI `0.963/0.971`
    - `multiwell_energy/energy_hd`: mean ARI `0.794/0.916`
    - `multiwell_strong_transition/strong_transition_hd`: mean ARI `0.931/0.918`
    - `generic_sparse` tends to have highest ARI; LISTA families close behind
  - **Duffing (2 basins): weak positive.** Mean ARI `0.19–0.24` across all views. Root cause: within-basin support consistency is only ~10% (basin 0: 12.8%, basin 1: 7.5%), so ~90% of trajectories activate a different support than their basin's mode. The encoder learned basin-discriminative continuous representations but not basin-aligned sparse supports on this system.
  - **Kuramoto (5 basins): negative.** Mean ARI ~`0` across all views. Supports are genuinely non-separable: within-basin vs between-basin Hamming distance ratio is 1.004 (flat). Winding-number basin distribution is highly imbalanced (q=0: 59%, q=±2: <1%). **Bug fix:** the v2 linear accuracy (~0.92–0.99 on support views) was a measurement artifact — singleton basins caused a fallback to train accuracy with no CV; corrected 3-class CV gives `0.427` (below majority baseline). Fix applied in `evaluate_label_free_clustering_v2.py`: singleton classes are now dropped before CV.
  - **Direct uniqueness update:** this v2 negative is strong evidence against basin recoverability, and the completed Kuramoto unique mode-support audit now directly confirms that Kuramoto also lacks meaningful literal basin-specific mode supports. Uniqueness is trivially degenerate (every trajectory has its own singleton support), so the Kuramoto negative is established on both the clustering and literal-support-uniqueness fronts.
  - **Feature view comparison:** `last_step_cosine` is strongest on multiwell; discrete support views (`majority_support`, `modal_support`) are competitive but do not clearly outperform PCA'd cosine features; `traj_mean_cosine` (v1 baseline) is comparable after PCA, suggesting v1 failure was primarily concentration-of-measure rather than averaging.
- **Interpretation:**
  - The multiwell positives (8/8 systems, ARI 0.71–1.00) upgrade the basin-support claim from "per-timestep uniqueness" to **"label-free basin recovery is possible"** without training-time basin labels. This is a key paper claim.
  - The kuramoto negative is genuine — supports carry zero basin-discriminative signal (flat Hamming geometry, every trajectory unique, ~93 active dims in all basins). This limits the label-free claim to potential-well systems.
  - The duffing weak result demonstrates that per-timestep support uniqueness (2/2) does not guarantee trajectory-level basin-support alignment when within-basin consistency is low (~10%). This is an honest limitation worth reporting.
- **Competitive LV multi-basin retrain and representation family (forecasting complete March 11; representation family complete March 13):** The previous `competitive_lv` benchmark was trivial (1 observed basin at `INTERACTION_SCALE=0.35`). The config is now `0.70`, producing 4 major basins. All 28/28 training tasks completed (arrays `8922033` + recovery `8922374`). Forecasting collection and comparison are done. All models are inside the good-forecast band; the paper-facing `200k` result is `generic_sparse=0.1254`, with `lista_blockdiag` and `lista_dense` close behind. The older `50k` comparison is only an auxiliary sanity check. Competitive LV is not a problematic forecasting holdout for any architecture. The related representation family is now also complete: support alignment on the corrected checkpoints remains weak/trivial, the repaired March 13 label-free clustering v2 rerun finished cleanly, and the final clustering read stays negative for stable support-defined basin recovery on the clean `4`-basin paper roots. Do NOT re-use any old 1-basin `competitive_lv` evaluation results.

## Decision Rules

- If default `dt` is poor and the benchmark rescue chain requests halving, prefer **smaller `dt`** before broader model changes or `10x` longer training.
- **The `200k` results are now the primary paper evidence.** The `v4` `50k` matrix is retained only as appendix / historical context for the matched four-model audit. All headline paper claims, cross-system rankings, and model comparisons should be drawn from the `200k` runs in `results/paper_followup_recipes_200k_20260309`.
- If a `200k` rerun materially improves over a `50k` result, do not foreground the `50k` result in the paper body.
- Use `generic_sparse_ns200k_best` as the primary paper anchor (not the `50k` `generic_sparse`).
- Use the promoted dense Stage-4 root (`lista_dense_promoted_stage4`) as the primary dense LISTA comparator.
- Use dense LISTA as the cross-system LISTA reference, but keep `lista_blockdiag` as the only LISTA-family candidate for intrinsic-HD follow-up unless new evidence clearly overturns that ranking.
- Treat the completed dense-LISTA easy-system Stage-1 sweep as evidence that external optimization alone can recover most easy dense-LISTA near-misses without changing architecture or `dt`.
- Treat the completed dense-LISTA Stage-2 / Stage-4 chain as the parity decision point:
  - stop coefficient-only holdout tuning
  - promote `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3` as the single fair dense recipe
  - use the completed Stage-4 rerun as the current dense parity evidence (`21/29` wins, `26/29` good systems, `0` dense-fails-anchor-passes systems)
  - when discussing the fair `200k` comparison, make the split explicit: dense wins more systems and keeps more systems good, while `generic_sparse_ns200k_best` has the best overall median
  - `v4` remains an appendix-only symmetric four-model audit; the `200k` follow-up is the primary paper-facing comparison
- Treat the full-benchmark block-diagonal dense-opt transfer as a negative result for global parity:
  - do not promote `lista_blockdiag_ns200k_denseopt_sc3em3` or `lista_blockdiag_ns200k_denseopt_sc6em3` as paper baselines
  - mention them only as targeted positives on `multiwell_strong_transition` / `multiwell_strong_transition_hd` or as evidence that the dense optimizer does not transfer cleanly to block-diagonal LISTA
- Treat historical block-diagonal claims through the repaired March 17/19 MLP controls only; the old mislabeled controls remain invalid provenance and should not be reused.
- Use the repaired intrinsic-HD rerun as the current decision-grade targeted evidence.
- Keep `evaluation_results_best.json` as the official checkpoint-selection rule for now, but treat `evaluation_results_last.json` as an important diagnostic on Kuramoto when discussing model-selection limits.
- Use the completed `dt=0.00625`, `200k` follow-ups and Kuramoto dimension sweep as the hard-system evidence:
  - on Kuramoto, emphasize that `lista_blockdiag` is robustly in-band at `N=16/24/32`, is not fully robust at `N=8`, and fails by `N=64`
  - on Hopfield, emphasize that smaller `dt` rescues periodic-reencoding forecasts for both models, but `generic_sparse` remains better
  - do not claim autonomous stability on the hard systems; every-step rollout errors remain the main limitation
- Treat the active Hopfield basin-count sweep as mechanism mapping only:
  - use it to test whether higher basin count changes the architecture ordering on Hopfield
  - do not use it to overwrite the canonical Hopfield paper claim unless the modified environment is explicitly framed as a new benchmark setting
- Use the completed Kuramoto dimension sweep to frame the hard-system claim:
  - claim a smaller-`dt` Kuramoto rescue for `lista_blockdiag` through `N=32`, not through `N=64`
  - make the `N=64` failure (`23.27`, `2/5` good seeds) and the non-robust `N=8` result (`8.11`, `4/5` good seeds) explicit
  - do not present promoted dense LISTA as a positive Kuramoto transfer result

## Highest-Value Audit Files

- [Current experiment log](/home/mila/l/lia/skae/docs/EXPERIMENTS.md)
- [Broad labelable-system support-alignment summary](/home/mila/l/lia/skae/results/paper_benchmark_support_alignment_20260311_v4_labelable/summary.md)
- [follow-up `200k` paper summary](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/paper_benchmark_summary.md)
- [follow-up `200k` forecasting summary](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/forecasting_summary.md)
- [fair `200k` `generic_sparse` vs canonical `generic_sparse`](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/compare/vs_canonical_generic_sparse/generic_sparse_ns200k_best_vs_generic_sparse/forecasting_comparison.md)
- [promoted dense Stage 4 vs fair `200k` `generic_sparse`](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/compare/vs_generic_sparse_ns200k_best/lista_dense_promoted_stage4_vs_generic_sparse_ns200k_best/forecasting_comparison.md)
- [Stage-4 dense rerun summary](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/collect/paper_benchmark_summary.md)
- [Stage-4 dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/compare/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3_vs_generic_sparse/forecasting_comparison.md)
- [Kuramoto dimension summary](/home/mila/l/lia/skae/results/kuramoto_dimension_sweep_dt00625_200k_20260309/collect/kuramoto_dimension_summary.md)
- [Stage-1 dense-LISTA easy-system summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/collect/paper_benchmark_summary.md)
- [Kuramoto recovery summary](/home/mila/l/lia/skae/results/kuramoto_recovery_seq8_20260305/forecasting_summary.md)
- [intrinsic-HD `dt` rescue rerun summary](/home/mila/l/lia/skae/results/intrinsic_hd_dt_rescue_20260308_rerun1/forecasting_summary.md)
- Appendix-only `50k` references:
  - [v4 paper summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/paper_benchmark_summary.md)
  - [v4 final forecasting summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/forecasting_summary.md)
  - [v4 pass-2 `dt` resolution summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/dt_resolution/pass2/dt_resolution.md)
  - [dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_dense_vs_generic_sparse/forecasting_comparison.md)
  - [block-diagonal vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md)
