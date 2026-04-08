# Claude Catalog Audit

Date: April 7, 2026

## Purpose

This note reconciles the implemented Claude transition-rich catalog under
[skae/claude_catalog](/home/mila/l/lia/skae/skae/claude_catalog) with the
saved validation artifacts under
[results/claude_catalog_validation](/home/mila/l/lia/skae/results/claude_catalog_validation).

The goal is not to generate more concepts. The goal is to answer a simpler and
more important question for the paper: what does the current worktree actually
support as a benchmark-ready implemented system set?

Status note (April 7 scope freeze):

This audit remains the source of truth for what is implemented and grounded,
but it is not the current experiment selector. Forward interpretability
experiments are restricted to the fixed `17`-system shortlist in
[docs/EXPERIMENTS.md](/home/mila/l/lia/skae/docs/EXPERIMENTS.md); within the
Claude catalog, the in-scope systems are `arrested_spiral`,
`cal_asymmetric_3`, `cal_high_cross_3`, `cal_hexagon_6`, `cal_octagon_8`,
`cal_pentagon_5`, `cal_square_4`, `checkerboard_potential`,
`duffing_triple_well`, `snic_multi`, `transition_routes_4`,
`var_depth_gradient_4`, `var_diamond_4`, and `var_l_shape_5`.

Companion audit figure:

- [claude_catalog_audit_atlas.png](/home/mila/l/lia/skae/docs/figures/claude_catalog_audit_20260407/claude_catalog_audit_atlas.png)
- [claude_catalog_audit_atlas.svg](/home/mila/l/lia/skae/docs/figures/claude_catalog_audit_20260407/claude_catalog_audit_atlas.svg)
- [claude_catalog_audit_atlas.pdf](/home/mila/l/lia/skae/docs/figures/claude_catalog_audit_20260407/claude_catalog_audit_atlas.pdf)
- [claude_catalog_strict_pass_gallery.png](/home/mila/l/lia/skae/docs/figures/claude_catalog_audit_20260407/claude_catalog_strict_pass_gallery.png)
- [claude_catalog_strict_pass_gallery.svg](/home/mila/l/lia/skae/docs/figures/claude_catalog_audit_20260407/claude_catalog_strict_pass_gallery.svg)
- [claude_catalog_strict_pass_gallery.pdf](/home/mila/l/lia/skae/docs/figures/claude_catalog_audit_20260407/claude_catalog_strict_pass_gallery.pdf)
- [claude_catalog_accepted_pass_gallery.png](/home/mila/l/lia/skae/docs/figures/claude_catalog_audit_20260407/claude_catalog_accepted_pass_gallery.png)
- [claude_catalog_accepted_pass_gallery.svg](/home/mila/l/lia/skae/docs/figures/claude_catalog_audit_20260407/claude_catalog_accepted_pass_gallery.svg)
- [claude_catalog_accepted_pass_gallery.pdf](/home/mila/l/lia/skae/docs/figures/claude_catalog_audit_20260407/claude_catalog_accepted_pass_gallery.pdf)

Companion priority-screen artifacts:

- [priority_screen_results.json](/home/mila/l/lia/skae/results/claude_catalog_priority_screen_20260407/priority_screen_results.json)
- [combined_screening_results.json](/home/mila/l/lia/skae/results/claude_catalog_priority_screen_20260407/combined_screening_results.json)

## Concrete audit result

- The current registry contains `112` implemented systems.
- The original saved fast-screen artifact
  [screening_results.json](/home/mila/l/lia/skae/results/claude_catalog_validation/screening_results.json)
  covered only `68` of them and had just `1` strict pass, `snic_multi`.
- A new priority-screen pass on `15` previously unscreened systems is now
  complete under
  [results/claude_catalog_priority_screen_20260407](/home/mila/l/lia/skae/results/claude_catalog_priority_screen_20260407).
- The combined grounded screen now covers `83` implemented systems.
- That leaves an unscreened backlog of `29` implemented systems.
- The combined accepted-pass pool is now `12` systems.
- Within that accepted pool, the strict-crossing core is `8` systems:
  `cal_triangle_3`, `cal_pentagon_5`, `cal_asymmetric_3`,
  `var_depth_gradient_4`, `var_diamond_4`, `var_l_shape_5`,
  `hybrid_state_dep_rot_5`, and `transition_routes_4`.
- The accepted-but-relaxed subset is `4` systems:
  `cal_hexagon_6`, `snic_multi`, `cal_square_4`, and `cal_star_5`.
- `41` screened systems now pass basin-count and occupancy gates.
- Within the `15`-system priority queue, `11` systems are now accepted under the
  fast-screen gate, `1` fails only on the crossing gate, `1` fails on occupancy plus crossing
  (`var_random_5a`), and `2` collapse to only `2` endpoint basins
  (`mixed_dynamics_triple`, `slow_fast_triple`).

## What this means

The current worktree does **not** substantiate the statement in
[claude_transition_rich_catalog.md](/home/mila/l/lia/skae/docs/planning/claude_transition_rich_catalog.md)
that `44 systems` are already confirmed passing.

The stronger interpretation now supported by the saved artifacts is:

- the catalog implementation effort is real and substantial,
- the tuned control family is not imaginary, because several of those systems
  now pass under the same strict fast-screen code used elsewhere,
- the screening coverage is still incomplete,
- the implemented catalog is now a plausible **small grounded benchmark pool**
  rather than only a speculative backlog,
- but it is still not a validated `44`-system benchmark packet.

The earlier `44 confirmed passing` line was therefore not completely detached
from reality, but it clearly mixed together a tuned polygon/variant pathway and
a much larger unsupported extrapolation.

## Why the older summary looks mixed

The current saved artifacts appear to come from at least two different
generation paths:

- [screening_results.json](/home/mila/l/lia/skae/results/claude_catalog_validation/screening_results.json)
  is the output of
  [fast_screen_catalog.py](/home/mila/l/lia/skae/tools/fast_screen_catalog.py),
  which uses the current strict/relaxed screening gates and reports `68`
  rows.
- [diverse_screening.json](/home/mila/l/lia/skae/results/claude_catalog_validation/diverse_screening.json)
  is a small `13`-system result packet for the tuned polygon/variant family.
- [benchmark_selection_15.png](/home/mila/l/lia/skae/results/claude_catalog_validation/gallery/benchmark_selection_15.png)
  and the older markdown summary are consistent with that tuned polygon/variant
  pathway, not with the saved `68`-row fast-screen audit.

So the current `claude_transition_rich_catalog.md` note mixes together:

- the larger implemented registry,
- the smaller tuned polygon/variant subset,
- and a benchmark-style selection narrative that is not backed by the current
  original `68`-row fast-screen artifact.

After the new priority screen, the tuned-family part of that story looks partly
real rather than wholly stale:

- `cal_triangle_3`, `cal_pentagon_5`, `cal_asymmetric_3`,
  `var_depth_gradient_4`, `var_diamond_4`, `var_l_shape_5`,
  `hybrid_state_dep_rot_5`, and `transition_routes_4` do survive the strict
  per-basin crossing gate,
- `cal_hexagon_6`, `snic_multi`, `cal_square_4`, and `cal_star_5` are still
  valid accepted systems, but only through the relaxed crossing gate,
- and `cal_octagon_8`
  remains the cleanest screened high-basin retune target.

## Current accepted-pass pool

These are the currently grounded implemented systems that satisfy the current
fast-screen acceptance rule. The `Crossing mode` column distinguishes systems
that keep every basin inside the strict `0.30-0.70` band from systems that only
survive through the relaxed crossing gate.

| System | B | Crossing mode | Crossing | Min occ | Why it matters |
| --- | --- | --- | --- | --- | --- |
| `cal_triangle_3` | `3` | `strict` | `0.500` | `0.260` | Cleanest minimal polygon control |
| `cal_pentagon_5` | `5` | `strict` | `0.500` | `0.170` | Mid-count polygon control now supported by the saved artifact |
| `cal_asymmetric_3` | `3` | `strict` | `0.530` | `0.280` | Asymmetry survives without breaking the benchmark gate |
| `var_depth_gradient_4` | `4` | `strict` | `0.400` | `0.200` | Interpretable asymmetric occupancy stress test that now clears the strict screen |
| `var_diamond_4` | `4` | `strict` | `0.610` | `0.230` | Strong rotated-separatrix benchmark candidate |
| `var_l_shape_5` | `5` | `strict` | `0.510` | `0.120` | Non-convex geometry that still stays paper-legible |
| `hybrid_state_dep_rot_5` | `3` | `strict` | `0.460` | `0.220` | One real hybrid mechanism already survives the strict screen |
| `transition_routes_4` | `4` | `strict` | `0.430` | `0.200` | Best grounded explicit route-choice benchmark in the implemented catalog |
| `cal_hexagon_6` | `6` | `relaxed` | `0.540` | `0.130` | Mid-high basin polygon control now clears the relaxed gate with cleaner routing than the old screened packet |
| `cal_square_4` | `4` | `relaxed` | `0.330` | `0.230` | Clean `4`-basin baseline with one basin slightly under the strict band |
| `cal_star_5` | `5` | `relaxed` | `0.550` | `0.160` | More interesting radial topology than a pure polygon, but one basin over-crosses |
| `snic_multi` | `3` | `relaxed` | `0.388` | `0.287` | Still worth keeping as the non-multiwell mechanistic outlier |

## Screened retune frontier

These are the most useful non-passing systems after the new priority screen.

| System | Status | Why it still matters |
| --- | --- | --- |
| `cal_octagon_8` | near miss | High-basin control sometimes approaches the relaxed gate, but repeated official screens still leave one weak basin and one overactive basin |
| `var_random_5a` | near miss | Random geometry is almost useful, but occupancy and one basin-crossing gate both miss |
| `duffing_triple_well` | legacy near miss | Strong `3`-basin candidate with one overactive basin |
| `neural_decision_3choice` | legacy near miss | Mechanistically richer than pure wells, but crossing remains too low |
| `rational_field` | interesting hold-out | Global crossing is strong, but per-basin behavior is still too uneven |

## Remaining priority unscreened backlog

After the completed `15`-system priority screen, the best remaining unscreened
targets are:

- `var_mixed_widths_5`
- `var_grid_2x2`
- `var_random_4a`
- `non_voronoi_basins`
- `hybrid_rotating_centers_3`
- `cal_high_cross_3`
- `cal_low_cross_4`

Those are now the best next-screen targets if we want to keep pushing the
implemented catalog before switching effort back to the conceptual elite
designs.

## Interpretation for the paper

- The conceptual inventory in
  [transition_rich_system_inventory_20260406.md](/home/mila/l/lia/skae/docs/planning/transition_rich_system_inventory_20260406.md)
  remains useful as a mechanism-design space.
- The implemented Claude catalog is no longer only a **retune and screening
  backlog**; it now contains a grounded `12`-system accepted pool, with an
  `8`-system strict-crossing core plus a small retune frontier.
- The immediate paper-quality move is now to use this audit to describe the
  status of the fixed chosen Claude subset rather than to reopen broad packet
  selection.
- The broader accepted pool, retune frontier, and conceptual inventory remain
  provenance/background, not the live experiment scope.

## Recommended next steps

1. Use the accepted-pool and retune-frontier labels in this audit to annotate
   the fixed chosen Claude subset.
2. If additional Claude-catalog runs are launched on this branch, restrict them
   to `arrested_spiral`, `cal_asymmetric_3`, `cal_high_cross_3`,
   `cal_hexagon_6`, `cal_octagon_8`, `cal_pentagon_5`, `cal_square_4`,
   `checkerboard_potential`, `duffing_triple_well`, `snic_multi`,
   `transition_routes_4`, `var_depth_gradient_4`, `var_diamond_4`, and
   `var_l_shape_5`.
3. Treat broader retuning and unscreened screening as optional background work,
   not the active branch default.
