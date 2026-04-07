# Restored Transition-Docs Comparison Note

Date: April 6, 2026

## Objective

Record the places where the restored copies of

- `docs/planning/transition_rich_basin_partition_plan_20260331.md`
- `docs/planning/chart_switching_transfer_system_plan_20260331.md`

differ from the versions I explicitly read earlier in this conversation.

## Scope note

This comparison is limited to the portions I previously opened and quoted while
working:

- the top `~140` lines of the transition-rich basin-partitioning plan
- the top `~120` lines of the chart-switching transfer-system plan

I cannot certify byte-level equality for the unseen remainder of either file,
because neither path has committed Git history in this repository and the lost
local drafts are no longer available.

## Concrete result

The restored files are structurally consistent with the earlier versions I read,
but they are not identical. The differences fall into three buckets:

1. explicit restoration notes added near the top of both files
2. light wording cleanups and heading-style changes
3. a small number of substantive wording changes, including one mathematical
   parameter change in the transfer-system plan

## Side-by-side differences

### `transition_rich_basin_partition_plan_20260331.md`

Current file:
- [transition_rich_basin_partition_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md)

| Earlier version I read | Current restored file | Difference type | Notes |
| --- | --- | --- | --- |
| No restoration note after the subtitle. | Restoration note added at lines `5-5`: [transition_rich_basin_partition_plan_20260331.md#L5](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md#L5) | Substantive provenance note | New text says the copy is reconstructed from surviving branch docs and may not be byte-identical. |
| Heading was `## 🎯 Objective`. | Heading is `## Objective`: [transition_rich_basin_partition_plan_20260331.md#L9](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md#L9) | Cosmetic | Emoji removed. |
| “...to a new branch centered on **label-light basin partitioning and classification**.” | “...to a branch centered on **label-light basin partitioning and classification** on deterministic toy systems.”: [transition_rich_basin_partition_plan_20260331.md#L11](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md#L11) | Mild wording change | Current version makes the deterministic-toy-system scope explicit earlier. |
| Question 2 said “Can we build **deterministic**, native-plot toy systems whose mechanics are simple enough to study carefully in the paper?” | Current Question 2 says “Can we build deterministic native-plot toy systems whose mechanics are simple enough to study directly in the paper?”: [transition_rich_basin_partition_plan_20260331.md#L16](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md#L16) | Cosmetic wording change | Same meaning. |
| “This branch keeps forecasting as supporting evidence...” | “Forecasting remains supporting evidence...”: [transition_rich_basin_partition_plan_20260331.md#L19](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md#L19) | Cosmetic wording change | Same meaning. |
| Heading was `## 🔒 Locked decisions`. | Heading is `## Locked decisions`: [transition_rich_basin_partition_plan_20260331.md#L21](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md#L21) | Cosmetic | Emoji removed. |
| Locked choice for toy-system strategy: “Reuse and extend `multiwell`, plus add one new deterministic toy-system family”. | Locked choice now says “Reuse and extend `multiwell`, plus add deterministic chart-switching families”: [transition_rich_basin_partition_plan_20260331.md#L27](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md#L27) | Mildly substantive | Current wording reflects the branch after `gated_transfer_linear` was added. |
| Follow-up bullet said `gated_transfer_linear` had the detailed mathematical specification and “acceptance summary” in the companion plan. | Current bullet says it has the detailed mathematical specification and “calibration policy”: [transition_rich_basin_partition_plan_20260331.md#L41](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md#L41) | Mild wording change | Slightly different emphasis. |
| Follow-up bullet said the lead open work was “the first trained-model screening pass on the three-system suite rather than more toy-system design.” | Current bullet says the lead open work is “trained-model screening and claim selection, not more toy-system invention.”: [transition_rich_basin_partition_plan_20260331.md#L42](/home/mila/l/lia/skae/docs/planning/transition_rich_basin_partition_plan_20260331.md#L42) | Substantive status update | Current file reflects a later branch state than the version I first read. |

### `chart_switching_transfer_system_plan_20260331.md`

Current file:
- [chart_switching_transfer_system_plan_20260331.md](/home/mila/l/lia/skae/docs/planning/chart_switching_transfer_system_plan_20260331.md)

| Earlier version I read | Current restored file | Difference type | Notes |
| --- | --- | --- | --- |
| No restoration note after the subtitle. | Restoration note added at lines `5-5`: [chart_switching_transfer_system_plan_20260331.md#L5](/home/mila/l/lia/skae/docs/planning/chart_switching_transfer_system_plan_20260331.md#L5) | Substantive provenance note | New text says the file is reconstructed and may not be byte-identical. |
| Objective paragraph ended: “This system is meant to become the flagship positive toy for the claim that forecast failures are concentrated at chart changes rather than inside a single locally linear regime.” | Current paragraph says: “This system is meant to complement the broader transition-rich branch by making the local-chart story legible in geometry rather than only in metrics.”: [chart_switching_transfer_system_plan_20260331.md#L18](/home/mila/l/lia/skae/docs/planning/chart_switching_transfer_system_plan_20260331.md#L18) | Substantive framing change | The current restored file is more cautious and less “flagship positive” than the earlier wording. |
| Heading was `## Why A New System Is Needed`. | Heading is `## Why a new system is needed`: [chart_switching_transfer_system_plan_20260331.md#L20](/home/mila/l/lia/skae/docs/planning/chart_switching_transfer_system_plan_20260331.md#L20) | Cosmetic | Sentence-case change only. |
| `gated_local_linear` bullet said it “does not yet produce meaningful transfer...” | Current bullet says it “does not produce meaningful transfer...”: [chart_switching_transfer_system_plan_20260331.md#L24](/home/mila/l/lia/skae/docs/planning/chart_switching_transfer_system_plan_20260331.md#L24) | Cosmetic wording change | Same scientific meaning. |
| `multiwell_strong_transition` bullet said its “current transition metric is endpoint-conditioned and does not isolate the chart-switching phenomenon as cleanly as we want for the main paper story.” | Current bullet says the metric “is endpoint-conditioned and does not isolate the chart-switching phenomenon as cleanly as desired for the strongest paper story.”: [chart_switching_transfer_system_plan_20260331.md#L25](/home/mila/l/lia/skae/docs/planning/chart_switching_transfer_system_plan_20260331.md#L25) | Cosmetic wording change | Same meaning. |
| Section ended with “This is the intended mechanistic complement to the existing forecasting packet.” | That sentence is gone. Current file instead adds a paper-facing caveat block at lines `42-45`: [chart_switching_transfer_system_plan_20260331.md#L42](/home/mila/l/lia/skae/docs/planning/chart_switching_transfer_system_plan_20260331.md#L42) | Substantive framing change | Current restored file now acknowledges that the strongest chart-localization claim was not supported on the learned-model pass. |
| Mathematical defaults listed “calibrated default: `R = 1.85`, `\\phi_0 = \\pi/4`”. | Current file lists “calibrated default: `R = 1.85`, `\\phi_0 = 0`”: [chart_switching_transfer_system_plan_20260331.md#L103](/home/mila/l/lia/skae/docs/planning/chart_switching_transfer_system_plan_20260331.md#L103) | Substantive mathematical change | This is the most important mismatch. The current restored value `\\phi_0 = 0` matches the implemented code default in [skae/data.py#L1793](/home/mila/l/lia/skae/skae/data.py#L1793). The earlier `\\pi/4` line appears to have been stale relative to the code. |

## Interpretation

The restored files are best understood as:

- faithful to the current branch structure and scientific role split
- not exact reproductions of the earlier drafts I previously read
- slightly updated toward the current paper-facing interpretation after the
  learned-model results were already known

The most important difference is not style. It is:

- the transfer-system plan is now more cautious about claiming a clean flagship
  chart-localization positive
- the transfer-system mathematical note now uses `\phi_0 = 0`, which matches
  the implemented environment and therefore is more trustworthy than the earlier
  `\pi/4` wording

## Project implication

If your goal is to recover the exact lost wording, these restored files should
be treated as close reconstructions, not exact copies.

If your goal is to have plan files that are consistent with the current branch
state and implementation, the restored files are better aligned than the older
wording in at least one important place: the transfer-system center phase.

## Next step

If you want an exact “best-effort reconstruction” of the earlier wording I saw,
the next step would be to create alternate snapshots, for example:

- `transition_rich_basin_partition_plan_20260331_reconstructed_from_conversation.md`
- `chart_switching_transfer_system_plan_20260331_reconstructed_from_conversation.md`

using only the earlier excerpts preserved in this conversation, without mixing
in later branch-state updates.
