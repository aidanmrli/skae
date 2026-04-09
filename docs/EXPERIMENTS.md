# Experiments (Core)

Date: April 8, 2026
Paper-critical live queue status last refreshed: `2026-04-08 23:21 EDT`

## Current Status Summary

Problem we are solving:
- Show on the fixed `17`-system interpretability shortlist that LISTA encoders
  separate endpoint basins into reusable sparse-support patterns, and that this
  basin separation is stronger than for a matched standard MLP encoder control.

Current paper-facing approach:
- Keep the live branch fixed to the `17` selected systems and no others:
  `multiwell_strong_transition`, `gated_local_linear`,
  `gated_transfer_linear`, `arrested_spiral`, `cal_asymmetric_3`,
  `cal_high_cross_3`, `cal_hexagon_6`, `cal_octagon_8`, `cal_pentagon_5`,
  `cal_square_4`, `checkerboard_potential`, `duffing_triple_well`,
  `snic_multi`, `transition_routes_4`, `var_depth_gradient_4`,
  `var_diamond_4`, and `var_l_shape_5`.
- Treat basin-support alignment as the primary branch objective. Forecasting,
  transition-path diagnostics, and oracle chart analyses remain secondary
  checks that help explain why a support partition is or is not trustworthy;
  they are not the main success criterion for this branch.
- The first live LISTA basin-partition sweep on the fixed `17` systems was
  launched at each system's default `dt` under
  [transition_rich_basin_partition_20260407](/network/scratch/l/lia/skae/transition_rich_basin_partition_20260407).
  That already-launched April 7 queue used the older `200000`-step budget and
  should now be treated as legacy long-budget LISTA evidence rather than the
  forward default.
- All new interpretability / metric-diagnosis training on the fixed
  `17`-system shortlist should use `20000` optimization steps. Reserve
  `200000`-step training for the final paper-facing rerun only, after the
  shortlist, metrics, and model recipe are locked.
- Use the standard MLP encoder control as the main causal comparator for this
  branch. The older broader zero-sparsity MLP mechanism packet remains useful
  supporting context, but the main branch-level claim still requires a matched
  fixed-`17` comparison reduced in the same basin-separation terms.
- Do not treat stronger chart-switch-localization or sparse-only mechanism
  claims as the acceptance criterion for this branch. Those questions remain
  useful supporting context, but the live branch succeeds or fails on whether
  LISTA yields cleaner basin-separated support structure than the matched
  standard MLP encoder on the fixed shortlist.
- The overnight April 8 continuation closes the practical step-size question
  more broadly than the earlier native-trio manual audit:
  [collect_pass0](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/collect_pass0/forecasting_summary.md)
  already reports `17/17` good systems at `H1000` for both dense and
  block-diagonal LISTA on the fixed shortlist, and the six completed overnight
  `20k` rerun waves keep all `16/16` rerun systems below the same gate at the
  same default per-system `dt`.
- Those overnight jobs should not be interpreted as evidence that smaller `dt`
  rescues were needed or helpful. The collector had been dropping `env_dt` for
  `gated_*` and `claude:*` arms, so the resolver kept reissuing the same
  default-`dt` tasks. The collector is now fixed locally; the scientific
  takeaway from the overnight work is stronger default-`dt` adequacy, not a
  smaller-`dt` effect.
- The first full fixed-`17` LISTA basin-support reduction is now complete
  under
  [basin_support_metrics_20260408_v3](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/basin_support_metrics_20260408_v3):
  - mean support-group purity is `0.9883` for block-diagonal LISTA and
    `0.9961` for dense LISTA
  - mean retained-trajectory coverage is `0.8729` for block-diagonal LISTA and
    `0.8787` for dense LISTA, with `15/17` systems above the `0.60` coverage
    gate for both roots
  - the local support-conditioned `H=20` fit does **not** beat the matched
    global fit on any of the `34` LISTA runs, and it beats the shuffled
    baseline only once (`dense x duffing_triple_well`) where coverage is only
    `0.1172`
- Exact metric definitions and caveats for that reduction now live in
  [docs/transition_rich_basin_support_metric_definitions.md](/home/mila/l/lia/skae/docs/transition_rich_basin_support_metric_definitions.md).
- The current LISTA packet therefore supports a basin-pure recurring-support
  claim much more strongly than a local-linear-mechanism claim. Even the best
  native positives now read as purity / coverage positives rather than as
  `H=20` local-linearity wins.
- The deterministic `2D` toy suite is locally implemented and calibrated:
  `multiwell_strong_transition`, `gated_local_linear`,
  `gated_transfer_linear`.
- Keep the fair `200k` benchmark packet, the hard-system packet, and the older
  mechanism packet only as supporting evidence rather than as the live branch
  definition.
- Historical design inventories, broader Claude-catalog screens, and older
  non-shortlist mechanism negatives are supporting provenance only. They are no
  longer the live branch definition.
- For design choices inside the next interpretability-ablation loop over plan
  items `3` and `4`, treat
  [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md)
  as the current ground-truth planning note, but do not treat those design
  choices as settled evidence until the corresponding experiments are run and
  written back into the live docs.

What stays live here:
- Fixed-`17` basin-separation evidence for dense-LISTA and block-diagonal
  LISTA.
- The matched standard-MLP control needed to test whether LISTA gives stronger
  basin separation than a non-LISTA encoder on the same systems.
- Native-trio and Claude-subset diagnostics that measure basin purity, support
  reuse, support-view clustering, and local predictive structure.
- Only the benchmark, hard-system, and older mechanism packets needed to
  position the branch in the broader paper.

Outstanding problem:
- No toy-system design blocker remains, and system selection is no longer
  open. The active interpretability branch is frozen to the `17` systems listed
  above. The full LISTA reduction now exists and says recurring supports are
  usually basin-pure and broad-coverage, but they do not yield a stronger
  support-conditioned `H=20` local-linear fit than one global map on this
  packet. The lead blocker is therefore no longer the LISTA reduction itself:
  it is the matched standard-MLP comparison on the same protocol together with
  a clean paper decision about how prominently to write the negative
  local-linearity read.

Assumption split:
- Training/deployment target: basin count and basin labels are unknown.
- Benchmark evaluation: known endpoint-basin counts and labels are allowed for diagnostics.

## Paper-Facing Experiment Protocol

1. Decide whether a result belongs in this live file before running or writing it down.
   - Keep it live only if it directly supports the benchmark, hard-system, or mechanism sections in `docs/review_main_results_tables_20260314.tex`, or if it is the newest paper-critical execution update.
   - Move appendix-only tuning, queue chronology, and superseded subthreads to `docs/EXPERIMENTS_ARCHIVE.md`.
2. Define the causal test before queueing.
   - Write the objective or claim, baselines and fairness controls, exact systems, seeds, horizons, metrics, acceptance criteria, failure criteria, and output roots.
   - For new toy systems, define `endpoint basin`, `transition`, basin-count target, crossing-fraction gate, deterministic mechanics, and calibration outputs before implementation.
3. Keep the paper fairness rules fixed.
   - Keep the frozen benchmark and hard-system supporting packets on the
     paper `200k` budget.
   - For the live interpretability / metric-design loop on the fixed
     `17`-system shortlist, use `20000` training steps by default and reserve
     `200000` only for the final confirmatory paper rerun after the shortlist,
     metrics, and model recipe are locked.
   - Report medians across seeds, not best-seed results.
   - Use the official checkpoint rule from `evaluation_results_best.json`.
   - Do not rely on basin labels or known basin counts when proposing training-time methods.
   - New deterministic toy systems must target `3-10` endpoint basins.
   - For the frozen first-pass pair, use the endpoint-conditioned crossing gate in the acceptable `0.30-0.70` range.
   - For the explicit chart-switching transfer family, use source-neighborhood transfer fractions together with inner-core retention rather than the old endpoint-conditioned crossing gate.
4. QA before queueing.
   - Run Python entry points with `uv run`.
   - Write and run the tests first before adding any system-specific environment code or metric code.
   - Smoke-test task builders and collectors locally.
   - Check task counts, seed coverage, root labels, line endings, and `sbatch --export` arguments before launch.
   - Submit SLURM jobs with `sbatch` on the `long` partition unless a different partition is explicitly justified.
5. Report results in this order.
   - Concrete result(s).
   - Result in experimental context.
   - Interpretation.
   - Project implications.
   - Next steps.
6. Update the docs in the same pass.
   - Refresh `Current Status Summary`, `Outstanding problems`, `Queue Status`, and the relevant live section in this file.
   - Update `docs/PAPER_TRACK_STATUS.md` whenever the result changes paper positioning, claims, or wrap-up priorities.
   - Archive detailed queue-era or lower-priority material in `docs/EXPERIMENTS_ARCHIVE.md` instead of letting this file grow again.

## Outstanding problems (active)

- The branch-wide LISTA basin-support reduction is now complete under
  [basin_support_metrics_20260408_v3](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/basin_support_metrics_20260408_v3).
  The main missing causal comparison is now the matched standard MLP encoder on
  the same fixed `17` systems and the same support-view protocol.
- The current April 7 live queue is LISTA-only. The matched standard MLP
  encoder control on the same fixed `17` systems is still the main missing
  causal comparison for the intended claim.
- The new LISTA reduction rules out the stronger reusable-local-linearity
  mechanism claim on the fixed shortlist:
  local `H=20` NRMSE loses to the global fit on all `34/34` LISTA runs, and
  beats the shuffled baseline only once, on a `0.1172`-coverage dense
  `duffing_triple_well` row. The live writing task is now to separate the
  purity / reuse claim from that negative mechanism read.
- The practical step-size question is no longer open on the fixed one-seed
  LISTA shortlist. Default `dt` already clears the user-facing
  `H1000 best-periodic < 50` gate on the full `17`-system packet for both
  LISTA roots, and the overnight continuation should be read as repeated
  default-`dt` `20k` reruns caused by the former `env_dt` collection bug
  rather than as true halved-`dt` evidence.
- The next live interpretability iteration should treat
  [docs/planning/basin_partition_experiments.md](/home/mila/l/lia/skae/docs/planning/basin_partition_experiments.md)
  as the ground-truth design inventory for the ablation axes attached to plan
  items `3` and `4`. Those axes still need systematic execution and reduction;
  until then, the note is design guidance rather than evidence.
- While that diagnostic loop is still live, do not spend new interpretability
  queue budget on `200k` runs. Use `20k` diagnostics first, then rerun the
  final locked paper packet at `200k`.
- The strongest current LISTA positives within the branch are now purity /
  coverage positives rather than local-linearity positives:
  `gated_local_linear` and `gated_transfer_linear` both keep pure retained
  groups with coverage `>= 0.74`, while `multiwell_strong_transition` remains
  coverage-limited (`0.3984` block-diagonal, `0.4531` dense).
- The native-trio manual audit already says the current LISTA failure mode is
  not support contamination: support groups remain basin-pure in every
  inspected native arm. The remaining free-rollout transition issues are still
  scientifically important, but they are secondary diagnostics for this branch
  rather than its primary acceptance criterion.
- The active Claude subset is fixed to `arrested_spiral`, `cal_asymmetric_3`,
  `cal_high_cross_3`, `cal_hexagon_6`, `cal_octagon_8`, `cal_pentagon_5`,
  `cal_square_4`, `checkerboard_potential`, `duffing_triple_well`,
  `snic_multi`, `transition_routes_4`, `var_depth_gradient_4`,
  `var_diamond_4`, and `var_l_shape_5`. Broader Claude-catalog screening is
  now provenance, not an open branch-selector.
- Historical non-shortlist mechanism negatives such as Kuramoto and corrected
  competitive Lotka-Volterra should remain supporting background rather than
  live acceptance criteria for this fixed-`17` branch.

## Recent Paper-Critical Result

### April 8, 2026: fixed-`17` LISTA basin-support reduction is purity-positive but local-linearity-negative

- Concrete result(s):
  the completed local reduction under
  [basin_support_metrics_20260408_v3](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/basin_support_metrics_20260408_v3)
  evaluated the latest fixed-shortlist LISTA checkpoints from
  [collect_pass6/forecasting_rows.csv](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/collect_pass6/forecasting_rows.csv)
  on the paper-facing recurring-support protocol. Across the `17 x 2 = 34`
  LISTA runs:
  - mean support-group purity is `0.9883` for
    `lista_blockdiag_basin_partition` and `0.9961` for
    `lista_dense_basin_partition`
  - mean retained-trajectory coverage is `0.8729` for block-diagonal LISTA and
    `0.8787` for dense LISTA
  - `15/17` systems clear the `0.60` coverage gate for both roots
  - local `H=20` NRMSE beats global `H=20` NRMSE on `0/17` systems for both
    roots
  - local `H=20` NRMSE beats the shuffled baseline on `0/17` block-diagonal
    systems and `1/17` dense systems; that lone dense win is
    `claude:duffing_triple_well`, where coverage is only `0.1172`
  - the main low-coverage exceptions are `claude:duffing_triple_well`
    (`0.1094` / `0.1172`) and `multiwell_strong_transition`
    (`0.3984` / `0.4531`)
- Result in experimental context:
  this is the first full fixed-shortlist reduction that matches the March
  recurring-support local-linearity protocol but applies it to the April 7/8
  LISTA shortlist packet rather than to the earlier three-system screening
  family. The reducer uses the latest collected row per
  `(root_label, system_key, seed)`, regenerates the shared held-out trajectory
  corpus from checkpoint-compatible environments, measures support-group purity
  and retained coverage, and compares local versus global versus shuffled
  `H=20` ridge fits.
- Interpretation:
  the LISTA shortlist now has strong evidence for basin-pure recurring-support
  structure on most systems, and the retained support groups are usually broad
  enough to survive the `0.60` coverage gate. But those same recurring
  supports do **not** currently justify the stronger statement that
  support-conditioned local linear maps explain held-out `20`-step dynamics
  better than a single global map.
- Project implications:
  the live branch can now be written as a basin-purity / recurring-support
  reuse result for LISTA, not as a local-linearity-mechanism win. That makes
  the matched standard-MLP control even more important: the next causal
  question is whether the same high-purity, high-coverage support structure is
  actually stronger for LISTA than for a non-LISTA encoder on the same packet.
- Next steps:
  run the matched standard-MLP control through the same reducer, add the
  resulting contrast to the fixed-`17` branch summary, and treat the new
  local-linearity numbers as a negative supporting diagnostic unless a later
  ablation changes that read.

### April 8, 2026: overnight rescue continuation broadened the default-`dt` LISTA read, but exposed an `env_dt` collection bug

- Concrete result(s):
  the six dependency-chained overnight rescue arrays
  `9193426`, `9193429`, `9193432`, `9193435`, `9193438`, and `9193441`
  completed between `2026-04-08 02:16 EDT` and `2026-04-08 20:24 EDT`,
  together with collectors / resolvers `9193427-9193443`, under
  [transition_rich_basin_partition_20260407](/network/scratch/l/lia/skae/transition_rich_basin_partition_20260407).
  These jobs were launched to halve `dt` whenever an arm failed
  `H1000 best-periodic < 50`, but the collected scientific read is different:
  - [collect_pass0](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/collect_pass0/forecasting_summary.md)
    already reports `17/17` good systems at `H1000` for both
    `lista_dense_basin_partition` and `lista_blockdiag_basin_partition` on the
    fixed shortlist
  - all `192/192` overnight rescue-pass training runs still satisfy
    `H1000 best-periodic < 50` at row level, even though they were supposed to
    be halved-`dt` rescues
  - across the rescue-only reruns, all `16/16` rerun systems remain below the
    same gate by system median for both LISTA roots
  - the final combined collector
    [collect_pass6](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/collect_pass6/forecasting_summary.md)
    still reports `17/17` good systems at `H1000`, with system-median
    `H1000 best-periodic` `0.0653` for dense LISTA and `0.0832` for
    block-diagonal LISTA
  - representative same-`dt` improvements across the repeated `20k` reruns are:
    dense `claude:checkerboard_potential` `0.3053 -> 0.0198`,
    dense `claude:cal_square_4` `0.0641 -> 0.00154`,
    dense `claude:var_l_shape_5` `0.0409 -> 0.00579`,
    block-diagonal `claude:var_l_shape_5` `0.9626 -> 0.0967`,
    block-diagonal `claude:cal_high_cross_3` `0.8760 -> 0.1727`,
    block-diagonal `claude:transition_routes_4` `0.5244 -> 0.1574`, and
    block-diagonal `gated_transfer_linear` `0.7945 -> 0.4899`
- Result in experimental context:
  the overnight chain was intended to test whether any fixed-shortlist LISTA
  arm needed smaller `dt` after the default-source collector ran. Instead, it
  exposed a collection / resolver plumbing issue:
  `32/34` rows in
  [collect_pass0/forecasting_rows.csv](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/collect_pass0/forecasting_rows.csv)
  and `224/226` rows in
  [collect_pass6/forecasting_rows.csv](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/collect_pass6/forecasting_rows.csv)
  were missing `env_dt`, even though the run configs stored the correct values.
  The cause was that
  [collect_forecasting_roots.py](/home/mila/l/lia/skae/tools/collect_forecasting_roots.py)
  only read built-in env `DT` fields and ignored
  `GATED_LOCAL_LINEAR`, `GATED_TRANSFER_LINEAR`, and `CLAUDE_CATALOG`.
  Because the resolver could not see those `dt` values, it kept reissuing the
  same default-`dt` task tables for the remaining arms rather than true
  halved-`dt` rescues.
- Interpretation:
  the scientifically relevant result is stronger default-`dt` evidence, not a
  smaller-`dt` rescue effect. On the current one-seed LISTA packet, smaller
  `dt` is not the practical bottleneck: the user-facing `H1000` gate is
  already satisfied on the full fixed shortlist for both dense and
  block-diagonal LISTA. The overnight `20k` reruns also suggest that the
  forward diagnostic recipe is already robust enough to stay inside that coarse
  forecasting gate on the same shortlist.
- Project implications:
  for the basin-separability branch, step-size rescue is now further
  de-emphasized. The live interpretability story should move even more clearly
  toward basin-support reduction and the matched standard-MLP comparison,
  because the overnight jobs neither open nor close the central causal claim;
  they only reinforce that default `dt` is already adequate on the one-seed
  LISTA shortlist. Operationally, do not spend more queue budget on pass `7`
  or further `dt` halving from this chain. The collector bug is fixed locally,
  so future collection will preserve `env_dt` for `gated_*` and `claude:*`
  arms.
- Next steps:
  treat the April 8 rescue chain as scientifically closed, recollect /
  re-resolve only if we want a clean selected-`dt` table after the collector
  patch, and spend the next queue budget on basin-separability evaluation and
  the matched standard-MLP control rather than on more `dt` rescue.

### April 7, 2026: initial default-`dt` LISTA basin-partition read on the native trio

- Concrete result(s):
  the first six manually evaluated native-system checkpoints from the live
  `17 x 2 x 1` LISTA sweep all already satisfy the `dt`-rescue gate at each
  system's default step size under `H1000 best-periodic < 50`.
  - `gated_local_linear`
    - dense: `H1000 best-periodic = 1.5437e-03`
    - block-diagonal: `H1000 best-periodic = 1.7160e-03`
  - `gated_transfer_linear`
    - dense: `H1000 best-periodic = 1.2213`
    - block-diagonal: `H1000 best-periodic = 6.7939e-01`
  - `multiwell_strong_transition`
    - dense: `H1000 best-periodic = 5.8041e-01`
    - block-diagonal: `H1000 best-periodic = 4.4335e-02`
  - the manual diagnostic root is
    [manual_eval](/network/scratch/l/lia/skae/transition_rich_basin_partition_20260407/manual_eval)
- Result in experimental context:
  these are checkpoint-by-checkpoint manual evaluations run on compute nodes
  while the live arrays are still training and before any natural
  `evaluation_best/.../rollout_artifacts.pt` outputs have appeared from the
  live sweep. The purpose was to test the new default-`dt` then halve-`dt`
  rescue rule early on the native trio, and to use the new rollout-diagnostics
  stack to identify failure modes beyond MSE.
  - `gated_local_linear`:
    both variants pass the gate, but `no_reencode` remains poor and predicted
    basin crossing fraction is `1.0` versus true `0.0`; block-diagonal raises
    basin endpoint accuracy from `0.03` to `0.34` but inflates basin
    transition-count MAE from `4.76` to `102.56`
  - `gated_transfer_linear`:
    both variants pass the gate; block-diagonal improves best-periodic `H1000`,
    basin transition-count MAE (`14.79` versus `30.12`), and support-switch
    count (`55.33` versus `303.64`), but both still predict basin crossing
    fraction `1.0` versus true `0.15`
  - `multiwell_strong_transition`:
    both variants pass the gate; block-diagonal sharply improves best-periodic
    `H1000` (`4.4335e-02` versus `5.8041e-01`), while dense is slightly better
    on basin transition-count MAE (`10.31` versus `12.89`) and support-switch
    count (`30.06` versus `33.93`)
- Interpretation:
  for the branch's main claim, the key positive result is that support groups
  remain basin-pure in every inspected native LISTA arm. Default `dt` is not
  the current blocker, and support contamination is not the current failure
  mode. The main remaining weakness is secondary: free rollouts still switch
  supports too often and invent too many basin crossings relative to the true
  native dynamics.
- Project implications:
  the paper should write this native-trio read around basin separation first:
  LISTA is already producing basin-pure support groups on the branch's most
  important native systems. Transition-fidelity diagnostics still matter, but
  they should be presented as a secondary limitation layered on top of an
  encouraging support-partition signal rather than as the branch headline.
- Next steps:
  wait for the first natural live-sweep `rollout_artifacts.pt` outputs, then
  extend the same support-view reduction across the Claude subset and the
  matched standard-MLP control. The next branch-level summary should count
  systems where LISTA gives cleaner basin separation than the non-LISTA
  baseline, not just systems with acceptable best-periodic error.

### April 7, 2026: implemented Claude-catalog audit

- Concrete result:
  [docs/planning/claude_catalog_audit_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_audit_20260407.md)
  now verifies that the current worktree has `112` registered Claude-catalog
  systems, `83` systems covered by the combined grounded fast screen, `29`
  implemented but unscreened systems, `12` accepted systems after a completed
  `15`-system priority screen, and an `8`-system strict-crossing subset within
  that accepted pool.
- Context:
  this audit was needed because the older
  [claude_transition_rich_catalog.md](/home/mila/l/lia/skae/docs/planning/claude_transition_rich_catalog.md)
  note claimed `44 confirmed passing` systems, which was strong enough to
  affect benchmark-positioning decisions if true.
- Interpretation:
  the implemented Claude catalog is real and useful, and it is no longer just a
  speculative backlog: several tuned controls and two hybrid mechanisms now
  survive the exact same fast screen. The retune pass also rescued
  `var_depth_gradient_4` into the strict-crossing core and `cal_hexagon_6`
  into the relaxed-accept subset. The stricter claim now needs one more split:
  `8` of the accepted systems keep every basin inside the strict crossing band,
  while `4` more survive through the relaxed crossing gate. Repeated official
  re-screens still leave `cal_octagon_8` outside the pool, so the high-basin
  control frontier is real rather than papered over.
  The
  saved artifacts still do not support treating the catalog as a validated
  large benchmark packet.
- Project implications:
  the paper should not promise a repaired `44`-system implemented benchmark.
  It can legitimately cite the grounded `12`-system Claude-catalog accepted
  pool, with an `8`-system strict core, as provenance for why the chosen
  Claude subset is technically grounded. But the audit no longer defines the
  active experiment scope by itself; forward branch experiments are restricted
  to the fixed `14` chosen Claude systems.
- Next steps:
  use the audit to annotate the chosen Claude subset by status, then plan or
  interpret runs only on the fixed `17`-system branch shortlist rather than
  reopening broader catalog or conceptual-inventory selection.

## Queue Status

- No live paper-critical forecasting or hard-system queue remains.
- One new local fixed-shortlist reduction is now complete without changing the
  queue:
  [basin_support_metrics_20260408_v3](/home/mila/l/lia/skae/results/transition_rich_basin_partition_20260407/basin_support_metrics_20260408_v3)
  summarizes support-group purity, retained coverage, and local/global/shuffled
  `H=20` NRMSE for the finished `17 x 2` LISTA packet.
- The legacy transition-rich LISTA queue is now scientifically resolved on its
  current one-seed scope:
  the default-source `200k` arrays finished, the six dependency-chained
  overnight `20k` rerun waves also finished, and the collected rows already say
  default `dt` clears the user-facing `H1000` gate on the full fixed shortlist
  for both LISTA roots.
- There is not yet a matched fixed-`17` standard-MLP branch queue in the live
  branch artifacts. That comparison remains the main missing causal control for
  the intended basin-separation claim.
- Forward training-budget policy for interpretability runs:
  all new queues on the fixed `17`-system shortlist should use `20000` steps,
  and `200000` should be reserved for the final locked paper rerun only.
- The active forward experimental scope is now frozen to `17` systems:
  `multiwell_strong_transition`, `gated_local_linear`,
  `gated_transfer_linear`, `arrested_spiral`, `cal_asymmetric_3`,
  `cal_high_cross_3`, `cal_hexagon_6`, `cal_octagon_8`, `cal_pentagon_5`,
  `cal_square_4`, `checkerboard_potential`, `duffing_triple_well`,
  `snic_multi`, `transition_routes_4`, `var_depth_gradient_4`,
  `var_diamond_4`, and `var_l_shape_5`.
- Default-`dt` sweep status:
  - launcher `9190857`: completed `0:0`
  - first array `9190869`: `16/34` tasks completed directly and `18/34` tasks
    failed and were rerouted
  - rerun array `9192341`: the `18` rerouted tasks all completed `0:0`
  - scratch result root:
    [transition_rich_basin_partition_20260407](/network/scratch/l/lia/skae/transition_rich_basin_partition_20260407)
- `dt`-rescue chain status:
  - launcher `9193402`: completed `0:0`
  - dependent collector/resolve chain `9193424-9193443`: completed `0:0`
  - dependent rescue arrays `9193426`, `9193429`, `9193432`, `9193435`,
    `9193438`, `9193441`: all completed their submitted work
  - the rescue task tables themselves were already rewritten to `20k`, but the
    completed rescue waves should be read as repeated default-`dt` reruns
    rather than as true halved-`dt` evidence because the collector had dropped
    `env_dt` on `gated_*` and `claude:*` rows
  - do not queue pass `7` from these artifacts; the scientific step-size
    question is already answered for this one-seed LISTA packet
- Natural live-sweep rollout artifacts are not available yet:
  no `evaluation_best/.../rollout_artifacts.pt` files have appeared under the
  live result roots so far.
- Manual native-system diagnostics already exist under:
  [manual_eval](/network/scratch/l/lia/skae/transition_rich_basin_partition_20260407/manual_eval)
- One new local catalog-audit pass is complete without changing the live queue:
  - audit note:
    [docs/planning/claude_catalog_audit_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_audit_20260407.md)
  - audit figure root:
    [docs/figures/claude_catalog_audit_20260407](/home/mila/l/lia/skae/docs/figures/claude_catalog_audit_20260407)
  - key result:
    the combined grounded Claude-catalog fast screen now supports `12`
    accepted systems across `83` screened systems, with an `8`-system
    strict-crossing core, not the stale `44 confirmed passing` headline in the
    older branch note
- No Claude-catalog training packet has been launched yet. The existing
  `claude_catalog_packet` manifest/task-builder/queue launcher now follow the
  same `20k` diagnostic budget, but they still encode the superseded
  `6`-system recommendation and should be treated as historical scaffolding
  rather than the active scope definition:
  - historical handoff / packet note:
    [docs/planning/claude_catalog_handoff_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_handoff_20260407.md)
  - historical manifest:
    [claude_catalog_packet_manifest.py](/home/mila/l/lia/skae/skae/benchmarks/claude_catalog_packet_manifest.py)
  - historical task builder:
    [build_claude_catalog_packet_tasks.py](/home/mila/l/lia/skae/tools/build_claude_catalog_packet_tasks.py)
  - historical compute-node queue launcher:
    [queue_claude_catalog_packet.sh](/home/mila/l/lia/skae/scripts/queue_claude_catalog_packet.sh)
- The completed transition-rich execution chain is:
  - Stage 1 array `9135303`: finished with `26/27` completed tasks and one
    failed cell, `9135303_20 = lista_dense_promoted_stage4 x multiwell_strong_transition x seed_2`
  - transition-rich collect job `9135304`: completed `0:0`
  - pairwise compare jobs `9135305`, `9135306`, `9135307`: completed `0:0`
  - post hoc chart-change attribution job `9135358`: completed `0:0`
  - post hoc support local-linearity job `9135411`: completed `0:0`
  - repo-side Stage 1 root:
    [results/transition_rich_screening_stage1_20260401](/home/mila/l/lia/skae/results/transition_rich_screening_stage1_20260401)
  - repo-side chart-attribution root:
    [results/transition_rich_chart_change_attribution_20260401](/home/mila/l/lia/skae/results/transition_rich_chart_change_attribution_20260401)
  - repo-side support-local-linearity root:
    [results/transition_rich_support_local_linearity_20260401](/home/mila/l/lia/skae/results/transition_rich_support_local_linearity_20260401)
  - scratch run root: `/network/scratch/l/lia/skae/transition_rich_screening_stage1_20260401`
- One new local benchmark-validity experiment is complete without changing the
  live queue:
  - oracle chart-switch collector root:
    [results/transition_rich_oracle_chart_switch_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_chart_switch_20260401)
  - execution/result note:
    [docs/planning/transition_rich_oracle_chart_switch_benchmark_plan_20260401.md](/home/mila/l/lia/skae/docs/planning/transition_rich_oracle_chart_switch_benchmark_plan_20260401.md)
- A second local benchmark-validity experiment is also complete without
  changing the live queue:
  - oracle refresh-cadence collector root:
    [results/transition_rich_oracle_refresh_cadence_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_refresh_cadence_20260401)
  - execution/result note:
    [docs/planning/transition_rich_oracle_refresh_cadence_plan_20260401.md](/home/mila/l/lia/skae/docs/planning/transition_rich_oracle_refresh_cadence_plan_20260401.md)
- The completed matrix was the full seed-robust Stage 1 screen, not a one-seed
  pilot:
  - `multiwell_strong_transition`, `gated_local_linear`,
    `gated_transfer_linear`
  - `generic_sparse_ns200k_best`, `generic_sparse_sc0_ns200k_best`,
    `lista_dense_promoted_stage4`
  - seeds `0,1,2`
- The March 25 seed-10 robustness backfill and the March 30 verification pass closed the paper-facing forecasting seed-coverage gaps used by `docs/review_main_results_tables_20260314.tex`.
- The March 31 raw-source seed-statistics reanalysis is now written in `docs/PAPER_SEED_STATISTICS_20260331.md`; it verifies raw-vs-collector agreement and records the remaining raw finite-value coverage gaps explicitly.
- The March 21 matched zero-sparsity MLP campaign is fully collected across benchmark, hard-system, and mechanism families.
- The March 17-20 repaired block-`K` fairness-control wave is fully collected and replaces the invalid historical MLP `+ block_diagonal K` rows.
- The tests-first transition-rich implementation pass is now complete locally:
  - the new tests pass
  - the new calibration tool exists at [tools/calibrate_transition_system.py](/home/mila/l/lia/skae/tools/calibrate_transition_system.py)
  - two secondary `2D` candidates are frozen from deterministic calibration
  - the explicit-transfer toy `gated_transfer_linear` is now
    implemented, calibrated, and plotted under
    [docs/figures/chart_switching_transfer_20260331](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331)
- The standardized evaluation stack now also contains a deterministic flow-branching diagnostic for the frozen `2D` systems, with threshold sweeps, ground-truth null baselines, and `2D` visual artifacts written through the usual checkpoint evaluation path.
- The new transition-rich screening tooling now also exists and passes local
  tests:
  - task builder:
    [tools/build_transition_rich_screening_tasks.py](/home/mila/l/lia/skae/tools/build_transition_rich_screening_tasks.py)
  - screening summary collector:
    [tools/summarize_transition_rich_screening.py](/home/mila/l/lia/skae/tools/summarize_transition_rich_screening.py)
  - SLURM wrappers:
    [scripts/collect_transition_rich_screening.sh](/home/mila/l/lia/skae/scripts/collect_transition_rich_screening.sh),
    [scripts/queue_transition_rich_screening_stage1_20260401.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_screening_stage1_20260401.sh)
  - focused validation:
    `uv run python -m pytest tests/test_transition_rich_screening_tasks.py tests/test_transition_rich_screening_summary.py tests/test_paper_followup_recipe_tasks.py -q` -> `4 passed`
- The post hoc chart-change attribution tooling for `gated_transfer_linear`
  now also exists and passes local tests:
  - plan:
    [docs/planning/transition_rich_chart_change_attribution_plan_20260401.md](/home/mila/l/lia/skae/docs/planning/transition_rich_chart_change_attribution_plan_20260401.md)
  - per-run evaluator:
    [tools/evaluate_transition_rich_chart_change_attribution.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_chart_change_attribution.py)
  - batch collector:
    [tools/collect_transition_rich_chart_change_attribution.py](/home/mila/l/lia/skae/tools/collect_transition_rich_chart_change_attribution.py)
  - SLURM wrappers:
    [scripts/collect_transition_rich_chart_change_attribution.sh](/home/mila/l/lia/skae/scripts/collect_transition_rich_chart_change_attribution.sh),
    [scripts/queue_transition_rich_chart_change_attribution_20260401.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_chart_change_attribution_20260401.sh)
  - focused validation:
    `uv run python -m pytest tests/test_transition_rich_chart_change_attribution.py tests/test_transition_rich_screening_tasks.py tests/test_transition_rich_screening_summary.py -q` -> `7 passed`
- The post hoc support local-linearity tooling for the transition-rich suite
  now also exists and passes local tests:
  - plan:
    [docs/planning/transition_rich_support_local_linearity_plan_20260401.md](/home/mila/l/lia/skae/docs/planning/transition_rich_support_local_linearity_plan_20260401.md)
  - per-run evaluator:
    [tools/evaluate_transition_rich_support_local_linearity.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_support_local_linearity.py)
  - batch collector:
    [tools/collect_transition_rich_support_local_linearity.py](/home/mila/l/lia/skae/tools/collect_transition_rich_support_local_linearity.py)
  - SLURM wrappers:
    [scripts/collect_transition_rich_support_local_linearity.sh](/home/mila/l/lia/skae/scripts/collect_transition_rich_support_local_linearity.sh),
    [scripts/queue_transition_rich_support_local_linearity_20260401.sh](/home/mila/l/lia/skae/scripts/queue_transition_rich_support_local_linearity_20260401.sh)
  - focused validation:
    `uv run python -m pytest tests/test_transition_rich_support_local_linearity.py tests/test_transition_rich_chart_change_attribution.py tests/test_transition_rich_screening_tasks.py tests/test_transition_rich_screening_summary.py -q` -> `10 passed`
- The explicit-transfer toy now also passes real training-entry smokes through the
  actual CLI:
  - full generic-sparse smoke with standardized evaluation:
    [runs/transition_rich_smoke/generic_sparse_gated_transfer/20260401-003500/20260401-003125](/home/mila/l/lia/skae/runs/transition_rich_smoke/generic_sparse_gated_transfer/20260401-003500/20260401-003125)
  - zero-sparsity MLP smoke:
    [runs/transition_rich_smoke/zero_sparse_gated_transfer/20260401-010000/20260401-004017](/home/mila/l/lia/skae/runs/transition_rich_smoke/zero_sparse_gated_transfer/20260401-010000/20260401-004017)
  - dense LISTA smoke:
    [runs/transition_rich_smoke/lista_dense_gated_transfer/20260401-010000/20260401-004017](/home/mila/l/lia/skae/runs/transition_rich_smoke/lista_dense_gated_transfer/20260401-010000/20260401-004017)
- A real-artifact validation of the new summary collector also exists on the
  generic-sparse transfer smoke under
  [results/transition_rich_smoke_gated_transfer_summary_20260401](/home/mila/l/lia/skae/results/transition_rich_smoke_gated_transfer_summary_20260401).
- A real-artifact validation of the chart-change attribution collector also
  exists on the same undertrained generic-sparse transfer smoke under
  [results/transition_rich_smoke_chart_change_attr_20260401](/home/mila/l/lia/skae/results/transition_rich_smoke_chart_change_attr_20260401).
- A real-artifact validation of the support local-linearity stage now also
  exists on the undertrained transfer smokes under
  [results/transition_rich_smoke_support_local_linearity_20260401](/home/mila/l/lia/skae/results/transition_rich_smoke_support_local_linearity_20260401),
  with collector summary in
  [collector/summary.md](/home/mila/l/lia/skae/results/transition_rich_smoke_support_local_linearity_20260401/collector/summary.md)
  and per-run outputs for `generic_sparse`, zero-sparsity MLP, and dense LISTA
  in
  [per_run](/home/mila/l/lia/skae/results/transition_rich_smoke_support_local_linearity_20260401/per_run).
- Do not queue another broad benchmark or hard-system rerun by default while
  the transition-rich Stage 1 screen is live.

## April 1, 2026: Oracle Chart-Switch Benchmark Validity Read

Concrete result(s):
- The local oracle benchmark is complete under
  [results/transition_rich_oracle_chart_switch_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_chart_switch_20260401).
- At `H20` on all starts, oracle chart switching beats a single global linear
  fit by:
  - `0.129974` on `gated_local_linear`
  - `0.0206368` on `gated_transfer_linear`
  - `0.0144757` on `multiwell_strong_transition`
- At `H20` on all starts, oracle chart switching beats fixed-chart local
  rollouts by:
  - `0.0219174` on `gated_local_linear`
  - `0.00237432` on `gated_transfer_linear`
  - `0.00539429` on `multiwell_strong_transition`
- The harder-subset read is the key discriminator:
  - `gated_local_linear` chart-switch starts: `+0.157948` vs global,
    `+0.0293535` vs fixed-chart
  - `multiwell_strong_transition` chart-switch starts: `+0.0593379` vs global,
    `+0.0245243` vs fixed-chart
  - `gated_transfer_linear` chart-switch starts: `+0.0208039` vs global,
    `+0.00258307` vs fixed-chart
  - `gated_transfer_linear` transfer starts: `-0.0557357` vs global,
    `+0.00993001` vs fixed-chart
  - `gated_transfer_linear` transition-region starts: `-0.0118927` vs global,
    `+0.0174968` vs fixed-chart

Result in experimental context:
- This was a ground-truth benchmark-validity experiment, not a learned-model
  benchmark.
- We fit affine state-space maps directly to held-out transitions from the true
  deterministic systems and compared:
  - one global linear map
  - one fixed local chart per rollout
  - oracle chart switching under the true chart schedule
  - shuffled chart switching with matched chart counts
- The plan and exact evaluator live in
  [docs/planning/transition_rich_oracle_chart_switch_benchmark_plan_20260401.md](/home/mila/l/lia/skae/docs/planning/transition_rich_oracle_chart_switch_benchmark_plan_20260401.md),
  [tools/evaluate_transition_rich_oracle_chart_switch.py](/home/mila/l/lia/skae/tools/evaluate_transition_rich_oracle_chart_switch.py),
  and
  [tools/collect_transition_rich_oracle_chart_switch.py](/home/mila/l/lia/skae/tools/collect_transition_rich_oracle_chart_switch.py).

Interpretation:
- `gated_local_linear` is the cleanest mechanistic chart-switching positive in
  the suite.
- `multiwell_strong_transition` is a weaker but real secondary positive, with
  gains concentrated on switching trajectories.
- `gated_transfer_linear` remains scientifically useful because the true chart
  partition matters and fixed-chart local models are worse than oracle
  switching, but it is not the cleanest proof that chart switching dominates:
  on the hardest transfer and transition-region starts, a single global linear
  fit is still competitive or better.

Project implications:
- The suite should stay three-system, but paper positioning must change.
- `gated_local_linear` should carry the clean main-text mechanistic claim.
- `gated_transfer_linear` should be written as the explicit-transfer stress test
  rather than as the only flagship proof of chart-switching linearizations.
- `multiwell_strong_transition` should remain in the paper as a weaker
  shared-corridor secondary toy.

Next steps:
- Compare the collected trained-model results directly against this oracle
  ranking.
- If `gated_transfer_linear` stays weak relative to the oracle benchmark after
  the collected model read, retune or demote it only if the paper still needs a
  stronger transfer-localization benchmark.

## April 1, 2026: Oracle Refresh-Cadence Benchmark Validity Read

Concrete result(s):
- The local oracle refresh-cadence benchmark is complete under
  [results/transition_rich_oracle_refresh_cadence_20260401](/home/mila/l/lia/skae/results/transition_rich_oracle_refresh_cadence_20260401).
- At `H20`, the recovered fraction of the oracle-vs-fixed improvement on all
  starts is:
  - `gated_local_linear`: `0.988651` at `c=2`, `0.90577` at `c=5`,
    `0.665843` at `c=10`
  - `multiwell_strong_transition`: `0.968223` at `c=2`, `0.858159` at `c=5`,
    `0.639087` at `c=10`
  - `gated_transfer_linear`: `0.88425` at `c=2`, `0.503731` at `c=5`,
    `0.0577701` at `c=10`
- The largest cadence still preserving at least `90%` of the oracle-vs-fixed
  gain on all starts is:
  - `gated_local_linear`: `c=5`
  - `multiwell_strong_transition`: `c=2`
  - `gated_transfer_linear`: only `c=1`
- The true held-out median chart dwell lengths are:
  - `gated_local_linear`: `171`
  - `multiwell_strong_transition`: `92`
  - `gated_transfer_linear`: `33`
- On the hardest `gated_transfer_linear` transfer starts, oracle periodic
  refresh still improves over stale fixed local charts but not over the global
  fit:
  - `c=2`: `gain_vs_fixed = +0.00866824`, `gain_vs_global = -0.0569974`
  - `c=5`: `gain_vs_fixed = +0.00483172`, `gain_vs_global = -0.0608339`

Result in experimental context:
- This was a ground-truth benchmark-validity experiment, not a learned-model
  benchmark.
- We fit affine state-space local maps on held-out trajectories from the true
  systems, then refreshed the true chart identity every `k` steps while holding
  it fixed between refreshes.
- The causal question was whether the benchmark itself supports a periodic
  re-encoding interpretation, not just the extreme endpoints of “never refresh”
  versus “refresh every step.”

Interpretation:
- `gated_local_linear` is the cleanest periodic-refresh benchmark in the suite.
  It preserves most of the oracle benefit even at moderate cadence and degrades
  smoothly as chart identity becomes stale.
- `multiwell_strong_transition` is still a real positive, but it needs faster
  refresh than `gated_local_linear`.
- `gated_transfer_linear` is much more cadence-sensitive and its relevant
  comparison is stale local charts, not global linear dominance. That makes it
  a strong stress test, not the cleanest mechanistic benchmark.

Project implications:
- The paper now has two independent ground-truth reads supporting the same role
  split:
  - `gated_local_linear` should carry the clean main-text mechanistic claim
  - `multiwell_strong_transition` should remain the weaker secondary positive
  - `gated_transfer_linear` should be written as the explicit-transfer stress
    test
- This also sharpens the trained-model decision rule: learned periodic gains
  should be compared against the oracle cadence ranking, not only against the
  oracle per-step chart-switch ranking.

Next steps:
- Use the collected Stage 1 and post hoc reads below as the active paper-facing
  evidence for the transition-rich branch.
- Treat `gated_local_linear` as the main-text mechanistic positive and
  `gated_transfer_linear` as the transfer stress test unless a new benchmark
  overtakes them.
- Do not claim chart-change localization on `gated_transfer_linear` from the
  current evidence.

## April 1, 2026: Collected Stage 1 Transition-Rich Screen

Concrete result(s):
- The full `200k` three-system Stage 1 screen is collected under
  [results/transition_rich_screening_stage1_20260401](/home/mila/l/lia/skae/results/transition_rich_screening_stage1_20260401).
- `26/27` runs completed. The only missing cell is
  `lista_dense_promoted_stage4` on `multiwell_strong_transition`, seed `2`
  (`9135303_20` failed).
- `gated_local_linear` is a clean positive for all three model roots:
  - `H1000` median best-periodic: `0.0890151` (`generic_sparse`),
    `0.000710212` (zero-sparsity MLP), `0.000783794` (dense LISTA)
  - `H1000` median no-reencode: `27651.5`, `2.86748`, `206.989`
- `gated_transfer_linear` is also a clean positive for all three model roots:
  - `H1000` median best-periodic: `1.7954`, `1.80077`, `1.09863`
  - `H1000` median no-reencode: `6.54401e+20`, `2.45499e+34`, `1.25639e+12`
- `multiwell_strong_transition` is not a clean forecasting positive:
  - `H100` median best-periodic stays very large across all three roots:
    `310.41`, `94.5872`, `180.29`
  - `H500` median best-periodic is catastrophically unstable:
    `3.12343e+20`, `2.70981e+13`, `3.05402e+19`
  - only isolated `H1000` medians become small on some roots/seeds, so the
    system does not provide a stable paper-facing positive at the current
    budget
- The flow-branching reference rate under the selected periodic evaluation also
  separates the suite:
  - `gated_transfer_linear`: near zero (`0`, `0`, `6.127e-05`)
  - `gated_local_linear`: low but nonzero (`0.0199912`, `0.02078`, `0.0631283`)
  - `multiwell_strong_transition`: much higher (`0.233598`, `0.179023`,
    `0.306753`)

Result in experimental context:
- This was the first seed-robust learned-model screen on the calibrated
  three-system suite:
  `3` systems x `3` model roots x `3` seeds at the paper `200k` budget.
- The official forecasting summary is
  [forecasting_summary.md](/home/mila/l/lia/skae/results/transition_rich_screening_stage1_20260401/collect/forecasting/forecasting_summary.md),
  and the branch-specific transition-rich summary is
  [transition_rich_screening_summary.md](/home/mila/l/lia/skae/results/transition_rich_screening_stage1_20260401/collect/transition_rich/transition_rich_screening_summary.md).
- The Stage 1 acceptance rule was never “win on every toy”; it was to find out
  which systems become clean positives, which remain hard, and whether periodic
  re-encoding matters at all on the chart-switching suite.

Interpretation:
- The learned-model read agrees with the oracle role split on the two strongest
  systems:
  `gated_local_linear` is the clean mechanistic positive and
  `gated_transfer_linear` is a strong transfer-oriented forecasting positive.
- The strongest robust empirical effect in the branch is periodic re-encoding
  versus stale latent rollouts, not explicit sparsity versus the matched
  zero-sparsity MLP.
- `multiwell_strong_transition` remains scientifically interesting, but it is
  too unstable to serve as a clean flagship positive at the current budget.

Project implications:
- The transition-rich branch is now model-backed rather than oracle-only.
- Main-text benchmark positioning should center:
  - `gated_local_linear` as the clean mechanistic positive
  - `gated_transfer_linear` as the hard explicit-transfer stress test
- `multiwell_strong_transition` should be written as a weaker secondary toy or
  demoted if we do not invest in a targeted follow-up.
- The zero-sparsity MLP is competitive enough that the branch should currently
  be written as a periodic re-encoding and local-partition story, not as a
  sparse-only win.

Next steps:
- Do not queue another broad screen by default.
- Decide whether the missing dense-LISTA `multiwell_strong_transition` seed is
  worth rerunning only after `multiwell_strong_transition`'s paper role is
  settled.
- Use the two post hoc reads below to decide whether the branch supports a
  stronger mechanistic claim than “periodic re-encoding prevents stale-chart
  failures.”

## April 1, 2026: Collected Chart-Change Attribution Read

Concrete result(s):
- The post hoc chart-change attribution pass is complete under
  [results/transition_rich_chart_change_attribution_20260401](/home/mila/l/lia/skae/results/transition_rich_chart_change_attribution_20260401).
- On `gated_transfer_linear`, all three model roots show positive periodic gain
  on chart-change windows, but that gain is consistently smaller than the gain
  on non-switch and stable subsets:
  - chart-change-window gain: `0.580229`, `2.5197`, `0.678105`
  - non-switch gain: `3.13667`, `18.4534`, `2.50009`
  - transition-region gain: `0.259326`, `0.591487`, `0.30107`
  - stable-region gain: `3.02077`, `17.7154`, `2.43167`
- All three localization deltas are negative for all three roots:
  - chart-change vs non-switch: `-2.55644`, `-15.9337`, `-1.82199`
  - transition vs stable region: `-2.76144`, `-17.0195`, `-2.14286`
  - transfer-switch vs all transfer trajectories: `-2.19871`, `-20.3617`,
    `-1.79909`

Result in experimental context:
- This stage was run after the Stage 1 screen using each run's selected
  best-periodic mode.
- It was designed to test a stronger causal claim than plain forecasting
  success:
  whether periodic re-encoding helps specifically when the active local chart
  changes on the explicit-transfer toy.

Interpretation:
- The current `gated_transfer_linear` learned-model read does not support that
  stronger localization claim.
- Periodic re-encoding helps on the transfer toy overall, but the gains do not
  concentrate where true chart changes occur.

Project implications:
- Do not write the current transition-rich branch as “re-encoding helps mainly
  at chart switches” based on `gated_transfer_linear`.
- The defensible claim is weaker and cleaner:
  periodic re-encoding matters on the transfer toy, but the benefit is not yet
  localized to true chart-change windows.

Next steps:
- Demote the chart-localization claim in the paper draft unless new evidence is
  collected.
- If that stronger causal claim is still scientifically central, design a
  cleaner benchmark or extend the attribution read to `gated_local_linear`
  before making it a headline statement.

## April 1, 2026: Collected Support Local-Linearity Read

Concrete result(s):
- The post hoc support local-linearity pass is complete under
  [results/transition_rich_support_local_linearity_20260401](/home/mila/l/lia/skae/results/transition_rich_support_local_linearity_20260401).
- `gated_local_linear` is the strongest partition-reuse positive:
  - retained trajectory coverage: `0.83391`, `0.923875`, `0.806228`
  - chart-switch coverage: `0.855491`, `0.907514`, `0.791908`
  - local/global/shuffled `H`-step NRMSE:
    `0.00667185 / 0.0196433 / 0.0534285`,
    `0.00294161 / 0.0127981 / 0.03636`,
    `0.00862393 / 0.0358176 / 0.125676`
- `gated_transfer_linear` is a clear but weaker positive:
  - retained trajectory coverage: `0.581315`, `0.653979`, `0.716263`
  - transfer coverage: `0.4`, `0.7`, `0.6`
  - local/global/shuffled `H`-step NRMSE:
    `0.0453574 / 0.0864248 / 0.15116`,
    `0.0411338 / 0.0843096 / 0.197936`,
    `0.0474659 / 0.0930806 / 0.172619`
  - weighted source-endpoint-pair purity stays high:
    `0.884146`, `0.883495`, `0.888889`
- `multiwell_strong_transition` is positive but weak on coverage:
  - retained trajectory coverage: `0.352941`, `0.418685`, `0.33564`
  - local/global/shuffled `H`-step NRMSE:
    `0.0188171 / 0.0395969 / 0.128213`,
    `0.017011 / 0.0314335 / 0.170924`,
    `0.0173628 / 0.037489 / 0.135196`

Result in experimental context:
- This stage asks whether recurring support groups define reusable local
  partitions on real trained checkpoints, not only whether a root forecasts
  well.
- The crucial controls are the global local-dynamics fit and the shuffled
  support assignment baseline.

Interpretation:
- The answer is yes across the suite:
  recurring support groups recover nontrivial local predictive structure and
  beat both global fits and shuffled assignments.
- The cleanest case is `gated_local_linear`, the transfer toy is intermediate,
  and `multiwell_strong_transition` is weakest.
- The matched zero-sparsity MLP is also strong on this diagnostic, often as
  strong as or stronger than the sparse root.

Project implications:
- The current mechanism claim should be about reusable local partitions, not
  about an effect that only appears with an explicit sparsity penalty.
- This is good paper evidence for interpretability and local predictive
  structure, but it does not currently support a sparse-only mechanism story.

Next steps:
- Write the mechanism result as a three-way comparison, not as a sparse-vs-dense
  binary.
- If a sparse-specific claim is still desired, it will need new evidence beyond
  the current support local-linearity summaries.

## Core Evidence Snapshot

### Transition-Rich Basin Partitioning

1. Concrete results:
- The new tests required by the transition-rich plan now exist and pass:
  - `tests/test_transition_rich_system_determinism.py`
  - `tests/test_transition_rich_endpoint_labels.py`
  - `tests/test_transition_rich_crossing_metrics.py`
  - `tests/test_transition_rich_env_registry.py`
  - `tests/test_transition_rich_calibration_schema.py`
- The local verification commands completed cleanly:
  - `env -u VIRTUAL_ENV uv run python -m pytest tests/test_transition_rich_system_determinism.py tests/test_transition_rich_endpoint_labels.py tests/test_transition_rich_crossing_metrics.py tests/test_transition_rich_env_registry.py tests/test_transition_rich_calibration_schema.py -q` -> `17 passed`
  - `env -u VIRTUAL_ENV uv run python -m pytest tests/test_env_interface.py tests/test_data.py -k 'make_env or multiwell' -q` -> `5 passed`
- The paper-facing trajectory figure now exists at [transition_rich_100_trajectories.svg](/home/mila/l/lia/skae/docs/figures/transition_rich_20260331/transition_rich_100_trajectories.svg), with companion `pdf/png/json` artifacts in [docs/figures/transition_rich_20260331](/home/mila/l/lia/skae/docs/figures/transition_rich_20260331).
- Short training-entry smokes also pass through the real training CLI:
  - `generic_sparse` on `multiwell_strong_transition`
  - `generic_sparse` on `gated_local_linear`
  - matched zero-sparsity MLP smoke on `gated_local_linear`
  - small LISTA smoke on `multiwell_strong_transition`
- Those runs write checkpoints and training plots under [runs/transition_rich_smoke](/home/mila/l/lia/skae/runs/transition_rich_smoke).
- The deterministic calibration tool now exists at [tools/calibrate_transition_system.py](/home/mila/l/lia/skae/tools/calibrate_transition_system.py).
- The two native-plot `2D` paper candidates are now frozen from deterministic calibration on a fixed `17x17` initial-condition grid with a `60`-step finite rollout, `100`-step long-rollout endpoint check, `10`-step extension, and `10`-step settling window:
  - `multiwell_strong_transition`
    - endpoint basins: `5`
    - endpoint distribution: `62 / 62 / 62 / 62 / 41`
    - per-endpoint crossing fractions: `0.500 / 0.500 / 0.500 / 0.500 / 0.585`
    - overall crossing fraction: `0.512`
    - label stability rate: `1.000`
  - `gated_local_linear`
    - endpoint basins: `3`
    - endpoint distribution: `103 / 97 / 89`
    - per-endpoint crossing fractions: `0.670 / 0.577 / 0.539`
    - overall crossing fraction: `0.599`
    - label stability rate: `1.000`
- The explicit-transfer toy is now implemented and calibrated on the fixed
  `17x17` grid with a `180`-step endpoint rollout and `15 + 15` settling /
  extension check:
  - `gated_transfer_linear`
    - endpoint basins: `3`
    - source-neighborhood counts: `9 / 9 / 9`
    - per-source-neighborhood transfer fractions: `0.333 / 0.444 / 0.333`
    - overall source-neighborhood transfer fraction: `0.370`
    - core counts: `3 / 2 / 2`
    - per-core retention fractions: `1.000 / 1.000 / 1.000`
    - overall core retention fraction: `1.000`
    - label stability rate: `1.000`
- Paper-facing transfer-system figure artifacts now exist:
  - [gated_transfer_linear_region_map.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_region_map.svg)
  - [gated_transfer_linear_chart_trajectories.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_chart_trajectories.svg)
  - [gated_transfer_linear_endpoint_trajectories.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_endpoint_trajectories.svg)
  - [gated_transfer_linear_transfer_summary.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_transfer_summary.svg)
- A trajectory-figure audit is now also documented in
  [gated_transfer_linear_trajectory_plot_audit_20260401.md](/home/mila/l/lia/skae/docs/planning/gated_transfer_linear_trajectory_plot_audit_20260401.md).
  It records that the original figure packet used a uniform `10x10` grid, that
  only `8/100` plotted starts lie in source neighborhoods, and that clearer
  ground-truth trajectory views were added:
  - [gated_transfer_linear_uniform_start_grid_10x10.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_uniform_start_grid_10x10.svg)
  - [gated_transfer_linear_ground_truth_trajectories_10x10_starts.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_ground_truth_trajectories_10x10_starts.svg)
  - [gated_transfer_linear_ground_truth_source_starts_17x17.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_ground_truth_source_starts_17x17.svg)
  - [gated_transfer_linear_dense_source_trajectories.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_dense_source_trajectories.svg)
  - [gated_transfer_linear_flow_field.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_flow_field.svg)
  The dense-source panel uses `144` starts inside the three source
  neighborhoods, and the flow-field panel makes clear that the line-like
  transport geometry is part of the ground-truth system itself rather than a
  sampling artifact.
- The `multiwell_gradient` reference under the same corridor-aware labeling is not yet the clean low-transition control:
  - endpoint basins: `5`
  - per-endpoint crossing fractions: `0.433 / 0.422 / 0.422 / 0.410 / 0.242`
  - overall crossing fraction: `0.401`
- Undertrained support-local-linearity smoke validation now also exists on the
  transfer toy under
  [results/transition_rich_smoke_support_local_linearity_20260401](/home/mila/l/lia/skae/results/transition_rich_smoke_support_local_linearity_20260401):
  - on the collected `generic_sparse` smoke, retained support groups cover
    `0.858` of all trajectories, `0.865` of chart-switch trajectories, and
    `0.700` of transfer trajectories
  - `H20` latent NRMSE is `0.0701` for local fits versus `0.1309` for the
    global baseline and `0.2018` for shuffled assignments
  - weighted endpoint purity is `1.000` and weighted source-endpoint-pair
    purity is `0.891`

2. Context:
- The benchmark, hard-system, and mechanism packets are now strong enough to act as supporting evidence while this new branch is screened.
- The calibration pass used only ground-truth environment dynamics and ground-truth region labels:
  - `multiwell_*` now exposes explicit basin cores plus a central transition corridor
  - `gated_local_linear` exposes explicit basin cores and gate sectors with exact local Jacobians
- No forecasting model was used to define or validate these labels.
- The 100-trajectory figure verifies that both frozen systems are native-plot `2D` systems with visually interpretable mechanics, not only numerically acceptable calibration summaries.
- The support-local-linearity numbers above are still tooling-validation reads
  on `5`-step smoke checkpoints, not paper-facing scientific claims; they are
  useful because they verify that the new post hoc stage is nondegenerate on
  real artifacts and that it can see transfer/chart-switch trajectories rather
  than collapsing to easy stable-only subsets.
- Updated interpretation:
  - `gated_local_linear` is the cleanest oracle-validated chart-switching toy
  - `multiwell_strong_transition` is the weaker shared-corridor toy
  - `gated_transfer_linear` is the explicit-transfer stress test rather than
    the sole mechanistic flagship

3. Interpretation:
- System design is now closed at the toy-environment level for the lead branch.
- The branch now has one clean mechanistic positive, one harder explicit-
  transfer stress test, and one weaker shared-corridor toy, all with stable
  endpoint labels and explicit region charts.
- The remaining open question is whether the learned supports on these systems yield reusable, label-light partitions that beat simple controls.
- `multiwell_gradient` should stay secondary until we decide whether we need to retune it into a genuinely low-transition control.
- The real training CLI accepts the new systems cleanly, so the screening trio can proceed without more environment/plumbing work.
- The training and analysis tooling is now strong enough to run a real
  seed-robust screening pass rather than another provisional pilot.
- The branch also now has a clean post hoc mechanistic tool for testing whether
  periodic gains localize near true chart changes, without contaminating the
  live Stage 1 training queue with mixed evaluation code.
- The branch now also has a clean post hoc support-local-linearity tool for
  testing whether recurring support groups define reusable local charts on the
  same deterministic grid.

4. Project implications:
- Environment design is no longer the rate-limiting step.
- The first model pass is now concretely defined as the full `27`-run Stage 1
  screen at the paper budget, with the oracle benchmark now fixing the expected
  role split:
  - `gated_local_linear` as the clean mechanistic positive
  - `gated_transfer_linear` as the transfer stress test
  - `multiwell_strong_transition` as the weaker secondary toy
- The clean next interpretive read is now also fixed:
  use the post hoc chart-change attribution diagnostic on finished
  `gated_local_linear` and `gated_transfer_linear` runs to test whether
  periodic gains are actually localized at true chart switches.
- A second post hoc interpretive read is also fixed:
  use the transition-rich support-local-linearity stage on finished runs to
  test whether recurring support groups cover transfer/chart-switch
  trajectories and beat global/shuffled local-dynamics baselines.
- Primary diagnostics should emphasize partition reuse, local predictive structure, and transition handling; basin-label metrics remain benchmark-only secondary reads.
- Forecast-side diagnostics should now explicitly include deterministic flow consistency: nearly identical full states should not branch to materially different futures except at the tolerance level already observed in the ground-truth simulator.
- Forecast sanity is now a post-freeze screening metric rather than a toy-system design gate.

5. Next steps:
- Use the completed collected summaries above as the active evidence for this
  branch rather than treating the screen as pending.
- Center the paper-facing transition-rich story on:
  - `gated_local_linear` as the clean mechanistic positive
  - `gated_transfer_linear` as the transfer stress test
  - `multiwell_strong_transition` as a weaker secondary toy unless a targeted
    follow-up rescues it
- Do not make the stronger chart-localization claim on the current
  `gated_transfer_linear` evidence.
- Write the support-local-linearity result as a reusable-partitions positive
  that is not sparse-specific under the current controls.
- Decide whether to rerun the single missing dense-LISTA `multiwell` seed only
  after the final paper role of `multiwell_strong_transition` is settled.

### Cross-System Forecasting

1. Concrete results:
- The live benchmark headline is the verified `29`-system, fair-`200k` packet under `results/paper_zero_sparse_benchmark_200k_20260321`, together with the aligned canonical Kuramoto packet under `results/kuramoto_dt0p01_200k_canonical_20260323`.
- Best-periodic cross-system medians at `H100/H500/H1000/H1500/H2000/H2500/H3000` are:
  - sparse MLP: `2.947e-4 / 0.0051 / 0.0240 / 0.0795 / 0.1123 / 0.1591 / 0.2201`
  - zero-sparsity MLP: `3.021e-4 / 0.0063 / 0.0353 / 0.1067 / 0.1351 / 0.1738 / 0.1951`
  - dense LISTA: `3.440e-4 / 0.0047 / 0.0250 / 0.0449 / 0.0627 / 0.0880 / 0.1039`
- Good-system counts are:
  - sparse MLP: `27/25/25/26/25/25/25`
  - zero-sparsity MLP: `27/26/26/27/27/27/27`
  - dense LISTA: `28/25/26/25/25/25/25`
- Under one fixed `periodic_100` cadence, late-horizon medians at `H1500/H2000/H2500/H3000` are:
  - sparse MLP: `0.1070 / 0.2140 / 0.3690 / 0.6148`
  - zero-sparsity MLP: `0.1729 / 0.4542 / 0.7041 / 0.9349`
  - dense LISTA: `0.0744 / 0.2688 / 0.4437 / 0.6508`

2. Context:
- This is the benchmark family retained in `docs/review_main_results_tables_20260314.tex`.
- The dense Stage 1-4 tuning chain is appendix-only provenance for fixing the dense comparator and no longer belongs in the live paper-facing evidence.

3. Interpretation:
- Sparse MLP is still the clean anchor at `H100` and `H1000`.
- Dense LISTA is median-best at `H500` and from `H1500` through `H3000`.
- The zero-sparsity MLP broadens coverage on some later horizons, but it is not the benchmark-median winner at any reported horizon.
- Under one fixed cadence, explicit sparsity is still materially better than the matched zero-sparsity MLP, but dense LISTA remains the late-horizon median winner.

4. Project implications:
- The benchmark claim must now be written as a genuine three-way result.
- Do not reduce the benchmark story to “sparse beats dense” or “dense beats sparse.”
- Use this packet as the supporting benchmark anchor while the transition-rich branch is built.

5. Next steps:
- Keep the fair `200k` packet as the supporting benchmark anchor.
- Keep the dense recipe-selection chain and the old `50k` audit in `docs/EXPERIMENTS_ARCHIVE.md`.

### Hard-System Forecasting

1. Concrete results:
- Kuramoto remains the clearest targeted structured result. At `dt=0.00625`, `lista_blockdiag` rescues the `N=16` and `N=32` settings relative to the sparse anchor, but the benefit is selective and does not survive uniformly across the full family.
- The repaired block-diagonal MLP fairness controls are now the valid source of truth:
  - uniform-spread `N=16`: `generic_sparse_blockdiag = 8.13 / 24.76 / 91.23 / 399.74 / 1724.45` at `H1000/H1500/H2000/H2500/H3000`
  - dimension sweep `H1000`, `N=8/16/24/32/64`: `10.61 / 6.51 / 5.79 / 5.16 / 208.54`
- The repaired `N=64` block-diagonal MLP row is a real seed-fragile limit, not a provenance bug: `2/5` good seeds and a seed-median `H1000 = 208.54`.
- Hopfield stays negative for LISTA-family claims. Smaller `dt` improves errors, but sparse MLP remains best through `H2000` at `N=16`, and the quarter-step `N=64` probe remains strongly negative.
- Corrected competitive Lotka-Volterra is no longer a forecasting holdout. The corrected `4`-basin row is negative for a clean block-structured win, and the fixed `8`-basin follow-ups split by `dt` rather than yielding one stable architecture reversal.

2. Context:
- This section merges the smaller-`dt` rescues, Kuramoto robustness checks, repaired fairness controls, Hopfield quarter-step probe, and corrected competitive-LV follow-ups into one hard-system family.
- The historical MLP `+ block_diagonal K` rows are invalid and must not be reused.

3. Interpretation:
- Smaller `dt` is the dominant hard-system lever, but it does not create a uniform LISTA success story.
- Block structure helps selectively on Kuramoto, fails at `N=64`, and does not rescue Hopfield.
- Removing the sparsity penalty can improve some hard-system rows, so the hard-system narrative must stay selective rather than global.

4. Project implications:
- Keep `lista_blockdiag` scoped to targeted Kuramoto claims.
- Present Hopfield as an MLP-better limitation case and corrected competitive-LV as a mixed forecasting result with weak mechanistic follow-through.
- Use this family as limitation/support evidence rather than as a live execution branch.

5. Next steps:
- Do not queue new broad hard-system sweeps by default.
- If a figure is needed, prioritize one that shows the Kuramoto horizon split and the `N=64` failure boundary explicitly.

### Basin-Support And Mechanism

1. Concrete results:
- The broad support audit and label-free clustering family are now stable enough to write as one system-dependent mechanism section:
  - multiwell: strong positive
  - Duffing: weak
  - Kuramoto: negative
  - Hopfield: continuous separation without reusable sparse supports
  - corrected competitive-LV: negative
- The direct Kuramoto support audit remains the key hard negative. Every family still shows essentially trajectory-unique supports, low basin consistency, flat Hamming geometry, and no reusable non-singleton mode supports.
- The matched zero-sparsity MLP does not fix Kuramoto support reuse.
- The recurring-support local-linearity study is partial on `multiwell_strong_transition` and negative on targeted smaller-`dt` Kuramoto:
  - multiwell retained coverage stays below the planned acceptance gate (`0.4141-0.4844` depending on root)
  - Kuramoto retains `0` recurring support groups across the evaluated root-seed cases
- Corrected competitive-LV stays negative-to-weak: no clean paper root clears the support-view gate, cosine separation remains negative, and clustering collapses to unstable discovered group counts rather than stable basin recovery.

2. Context:
- This family is retained because it directly supports the basin-support alignment goal and the mechanism section in `docs/review_main_results_tables_20260314.tex`.
- Binary support uniqueness is no longer a paper headline metric because it saturates even on negative systems.

3. Interpretation:
- Explicit sparsity acts as a representation-shaping prior on some systems, especially multiwell, but it does not yield a universal basin-support win.
- Kuramoto remains the decisive negative result for support-defined regime reuse, and that negative survives the matched zero-sparsity control.
- Corrected competitive-LV narrows rather than broadens the mechanism claim.

4. Project implications:
- Write the mechanism section as system-dependent and bounded.
- Keep cosine separation, label-free recovery, and recurring-support local-linearity ahead of binary uniqueness in the paper narrative.
- Use this family as the metric starting point for the new transition-rich branch rather than as the branch endpoint.

5. Next steps:
- If one more mechanism analysis is needed, make it a small offline sensitivity or a targeted seed extension on the undercovered corrected competitive-LV support-alignment branch.
- Do not reopen broad representation sweeps by default.

## Recent Paper-Critical Updates

### March 31 Tests And Calibration Freeze Two Transition-Rich 2D Candidates

1. Concrete results:
- The transition-rich implementation pass is complete locally:
  - the required tests exist and pass
  - the new calibration module is in [skae/transition_calibration.py](/home/mila/l/lia/skae/skae/transition_calibration.py)
  - the reproducible calibration entry point is in [tools/calibrate_transition_system.py](/home/mila/l/lia/skae/tools/calibrate_transition_system.py)
- The paper-facing 100-trajectory figure now exists at [transition_rich_100_trajectories.svg](/home/mila/l/lia/skae/docs/figures/transition_rich_20260331/transition_rich_100_trajectories.svg).
- Short training-entry smokes also completed cleanly under [runs/transition_rich_smoke](/home/mila/l/lia/skae/runs/transition_rich_smoke).
- Two native-plot `2D` systems are now frozen as paper candidates from deterministic calibration on the fixed `17x17` grid:
  - `multiwell_strong_transition`: `5` endpoint basins, crossing fractions `0.500 / 0.500 / 0.500 / 0.500 / 0.585`, overall crossing `0.512`, label stability `1.000`
  - `gated_local_linear`: `3` endpoint basins, crossing fractions `0.670 / 0.577 / 0.539`, overall crossing `0.599`, label stability `1.000`
- The same calibration shows that `multiwell_gradient` is not yet the desired low-transition control under the new corridor-aware labeling (`overall crossing = 0.401`, center-basin crossing `0.242`).

2. Context:
- This is the first time the branch has concrete deterministic systems that satisfy the basin-count and transition-richness gates before any model training.
- The endpoint labels were checked only with ground-truth rollouts (`100` steps plus a `10`-step extension), not with model predictions.

3. Interpretation:
- The toy-system design problem is now solved for the positive pair.
- The lead blocker has shifted to model screening and diagnostic readout rather than further environment implementation.

4. Project implications:
- The branch can move directly to the screening trio and the label-light partitioning diagnostics.
- Forecasting becomes a supporting post-freeze read rather than the criterion used to design the systems.
- No additional environment integration work is needed before those screening runs.

5. Next steps:
- Train the screening trio on the two frozen systems.
- Add the sparse-anchor forecast sanity metric back into the calibration summary once the first sparse-anchor checkpoints exist.

### March 31 Paper Direction Shifts To Transition-Rich Basin Partitioning

1. Concrete results:
- The live paper direction is now frozen in [docs/planning/transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md).
- The fair `200k` benchmark packet, the hard-system packet, and the current mechanism packet are now treated as supporting evidence rather than the lead live branch.
- The new lead branch is deterministic transition-rich basin partitioning and classification on analyzable toy systems.
- The first-pass constraints are fixed: tests first, native-plot `2D` first, `3-10` endpoint basins, and acceptable per-endpoint-basin crossing fractions in the `0.30-0.70` range.

2. Context:
- This shift follows closure of the broad forecasting queue and stabilization of the current mechanism read.
- The remaining paper leverage is no longer another benchmark rerun; it is a cleaner narrative driver with controlled deterministic dynamics.

3. Interpretation:
- The project is no longer blocked on collecting more broad forecasting rows.
- It is blocked on building a toy-system branch that lets us study partition reuse, transitions, and forecasting failure modes mechanistically.

4. Project implications:
- Immediate work should move to tests, minimal interface scaffolding, candidate calibration, and the first screening trio.
- MSE becomes supportive rather than sufficient; the main new metrics should diagnose why a model works or fails.

5. Next steps:
- Implement the plan in order: tests, scaffolding, system dynamics, calibration, candidate freeze, and screening.
- Keep the existing benchmark, hard-system, and mechanism packets fixed while this new branch is built.

### March 31 Raw-Source Seed-Statistics Reanalysis Adds Robust Seed Summaries

1. Concrete results:
- A new raw-source report now exists at `docs/PAPER_SEED_STATISTICS_20260331.md`, with machine-readable tables under `results/paper_seed_statistics_20260331`.
- The reanalysis recomputes paper-facing results from the lowest-level surviving per-seed artifacts: `evaluation_results_best.json`, `support_alignment.json`, `analysis_results.json`, and local-linearity `metrics.json`.
- Raw-vs-collector verification passed across every family where both sources exist, including `27,060` checked label-free clustering cells and `12,180` checked benchmark best-periodic forecasting cells.
- The raw finite-value coverage audit is stricter than the earlier collector-row audit: benchmark best-periodic stays clean, but fixed-cadence late-horizon Hopfield and embedded-multiwell rows drop below `10` finite seeds, corrected `4`-basin CLV block-diagonal LISTA drops to `14` finite seeds at `H1000-H3000`, repaired Hopfield `N=64` block-diagonal MLP drops to `7` finite seeds at `H1500-H3000`, the direct Kuramoto mode-support audit stays at `5` seeds, and corrected competitive-LV support alignment stays at `3`.

2. Context:
- This pass was triggered to replace middle-seed medians with mean, standard deviation, IQM, and bootstrap confidence intervals computed directly from the original per-seed JSON outputs wherever those still exist.

3. Interpretation:
- The robust summaries are now provenance-clean.
- They also show that “collector row exists” and “finite seed statistic exists” are not always the same statement on the late-horizon tails.

4. Project implications:
- Use `docs/PAPER_SEED_STATISTICS_20260331.md` for any discussion that needs average-case rather than median-only reporting.
- Do not describe the entire paper-facing packet as uniformly `10`-seed clean without the raw finite-value caveat.

5. Next steps:
- Keep the new report as the seed-statistics companion to the senior-coauthor packet.
- If any coverage repair is queued, prioritize only the remaining mechanism gaps or the clearly identified late-horizon finite-value holes.

### March 30 Verification Pass Closes Forecasting Seed-Coverage Gaps

1. Concrete results:
- The table-facing forecasting packet used by `docs/review_main_results_tables_20260314.tex` is now seed-clean: every displayed benchmark and hard-system forecasting row uses at least seeds `0-9`, and corrected `4`-basin competitive Lotka-Volterra uses seeds `0-14`.
- The residual Kuramoto repair chain `9074821 -> 9074822 -> 9074823` filled the last missing forecasting seed (`generic_sparse_sc0_n8`, seed `6`).
- The March 31 raw-source reanalysis refined this collector-row picture: the remaining mechanism undercoverage is corrected competitive-LV support alignment (`0,1,2`) plus the direct Kuramoto mode-support audit (`5` seeds), and several late-horizon forecasting slices still fall below full finite-value coverage.

2. Context:
- This pass was a direct verification against the exact rows cited in the senior-coauthor handoff tables.

3. Interpretation:
- Forecasting coverage is no longer an execution blocker.
- The remaining undercoverage sits in a mechanism artifact, not in the paper-facing forecasting packet.

4. Project implications:
- The paper can now cite the forecasting packet as seed-clean.
- The remaining blocker is narrative accuracy, not missing forecasting seeds.

5. Next steps:
- Keep any future coverage work narrowly targeted to the undercovered corrected competitive-LV mechanism branch.

### March 21-23 Matched Zero-Sparsity MLP Campaign Closes Across Every Paper Family

1. Concrete results:
- The matched zero-sparsity MLP baseline is fully collected across benchmark, hard-system, and mechanism families under:
  - `results/paper_zero_sparse_benchmark_200k_20260321`
  - `results/zero_sparse_hard_systems_20260321`
  - `results/zero_sparse_mechanisms_20260321`
- The repaired zero-sparsity Kuramoto support-audit rerun confirms that removing explicit sparsity does not rescue reusable Kuramoto basin supports.

2. Context:
- This is the direct causal control for whether the sparsity penalty itself matters, because it keeps the same MLP encoder/decoder family and optimizer recipe as the sparse anchor while setting `lambda_sparse = 0`.

3. Interpretation:
- The zero-sparsity control materially changed the paper position.
- The paper can no longer present a simple sparse-versus-dense story, and it also cannot present zero sparsity as a new global winner.

4. Project implications:
- Keep the zero-sparsity MLP in the benchmark, hard-system, and mechanism tables as a first-class paper control.

5. Next steps:
- Use the three-way sparse-vs-zero-sparse-vs-LISTA framing consistently across all paper-facing docs.

### March 17-20 Repair Wave Replaces The Invalid Historical Block-`K` MLP Controls

1. Concrete results:
- A March 17 audit showed that the historical MLP `+ block_diagonal K` controls were invalid because `GenericKM` ignored `K_STRUCTURE` and always learned a dense latent transition.
- The repair wave completed cleanly and now provides the valid block-`K` MLP controls used in the hard-system family and the senior-coauthor packet.
- The March 19 retry wave closed the remaining Kuramoto mirror gaps, and the March 20 audit confirmed that the bad `N=64` repaired row is a real seed-fragility pattern rather than a stale artifact.

2. Context:
- These controls matter because they isolate whether a reported gain comes from encoder structure or only from imposing block structure on the latent transition.

3. Interpretation:
- Historical MLP `+ block-K` rows remain invalid provenance and should stay retired.
- The repaired controls show a selective picture: real positives on some Kuramoto settings, negative or mixed results elsewhere.

4. Project implications:
- All causal language about block structure must use the repaired March 17-20 outputs rather than the earlier historical rows.

5. Next steps:
- Keep the repaired controls as the only paper-facing source for block-`K` fairness claims.

## Archive Pointer

- `docs/EXPERIMENTS_ARCHIVE.md` now holds the appendix-only dense tuning chain, the old `50k` benchmark audit, superseded queue snapshots, intermediate CLV and Kuramoto follow-up chronology, and other lower-priority paper-era detail that no longer belongs in the live file.
