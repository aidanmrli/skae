# Transition-Rich System Catalog — Handoff

Date: 2026-04-07  
Branch: `claude-transition-rich-systems-gen`  
Prior conversation produced 9 commits on this branch.

## What was built

A transition-rich system catalog in `skae/claude_catalog/` for benchmarking
basin-aware Koopman autoencoders.

The branch populated eight catalog modules plus shared base/registry code. The
important grounded count is the registry-backed one from
[claude_catalog_audit_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_audit_20260407.md):

- `112` implemented registered systems in the current worktree
- `83` systems covered by the combined grounded fast screen
- `29` implemented systems still unscreened
- `12` currently accepted systems under the official fast-screen rule
- `8` systems in the strict-crossing core

Each benchmarkable system inherits from `CatalogSystem` (defined in
`skae/claude_catalog/base.py`) with `dynamics()`, `reset()`, and `step()`
methods using RK4 integration on `torch.float64` tensors. A decorator-based
registry (`skae/claude_catalog/registry.py`) maps string names to classes.

The standard training stack can now also access these systems through
`--env claude:<system_name>` rather than only through the separate screening
tools.

Read this as a grounded small benchmark pool plus a retune/screening frontier,
not as a validated large benchmark packet.

## Status note for the current branch

This document no longer defines the active follow-up packet for the branch.
Forward interpretability experiments are now restricted to a fixed `17`-system
shortlist:

- native transition-rich trio:
  `multiwell_strong_transition`, `gated_local_linear`,
  `gated_transfer_linear`
- Claude-catalog subset:
  `arrested_spiral`, `cal_asymmetric_3`, `cal_high_cross_3`,
  `cal_hexagon_6`, `cal_octagon_8`, `cal_pentagon_5`, `cal_square_4`,
  `checkerboard_potential`, `duffing_triple_well`, `snic_multi`,
  `transition_routes_4`, `var_depth_gradient_4`, `var_diamond_4`,
  `var_l_shape_5`

Use this handoff for implementation/audit context only. The older `6`-system
packet recommendation later in this file has been superseded as the live branch
scope.

### Module layout

| Module | Branch role |
|--------|-------------|
| `systems_gradient.py` | Gaussian-well gradient flows and early rotation variants |
| `systems_bio_physical.py` | Toggle-switch, Duffing, Josephson, cusp, and related physical/biological systems |
| `systems_creative.py` | SNIC, heteroclinic, maze, multi-scroll, Brusselator, and other more novel mechanisms |
| `systems_novel.py` | Waddington, protein-folding, climate-tipping, neural-decision, and related themed systems |
| `systems_tuned.py` | Calibrated Gaussian-well controls with independent rotation and route variants |
| `systems_variants.py` | Diverse well geometries such as diamond, L-shape, and depth-gradient layouts |
| `systems_hybrid.py` | Hybrid mechanisms that mix well templates with state-dependent transport terms |
| `systems_flagship.py` | Paper-motivated challenge systems such as Voronoi mismatch and damping-gradient variants |

Branch-provenance class counts across those files sum to more than the current
registry-backed `112`, so use the audit count above rather than file-level
counts when discussing benchmark readiness.

### Validation tools

- `tools/fast_screen_catalog.py` — The official screening pipeline. Uses `torch.vmap` for
  trajectory generation, DBSCAN/k-means for basin identification, per-basin crossing
  fraction computation. Saves results to JSON. **This is the source of truth for pass/fail.**
- `tools/validate_claude_catalog.py` — Older, slower full validation (largely superseded).
- `tools/plot_catalog_gallery.py` — Publication-quality gallery and true basin map generation.

### Acceptance gates (from `docs/planning/transition_rich_basin_partition_plan_20260331.md`)

| Gate | Requirement |
|------|-------------|
| Determinism | Same seed → same trajectory |
| Basin count | 3–10 endpoint basins |
| Basin occupancy | No basin below max(0.05, 0.5/B) |
| Crossing fraction | **Per-basin** crossing in [0.30, 0.70] (strict) |

The per-basin crossing check is the hardest gate. It requires that for EVERY basin,
30–70% of trajectories that end in that basin visited a different basin's region during
their transient. The overall crossing can be in range while individual basins fail.

## Key scientific finding

**Independent rotation** `dx/dt += ω·y, dy/dt -= ω·x` (constant angular velocity
independent of the gradient) is the mechanism that produces transition-rich dynamics.

- **Proportional rotation** (α·R·∇V, rotating the gradient) gives < 15% crossing —
  insufficient because it vanishes near saddle points where transitions should happen.
- **Independent rotation** with `ω/amp ∈ [0.3, 0.7]` gives 30–65% crossing for
  Gaussian well systems. This creates **spiral basin boundaries** that are fundamentally
  different from Voronoi cells.
- For 3–5 well systems on polygons, `ω ≈ 1.0, amp ≈ 2.0–3.0` is the sweet spot.
- For 6+ wells, even-numbered polygons have windward/leeward asymmetry from uniform
  rotation that makes per-basin gates very hard to pass. Odd-numbered (5, 7) are easier.

## Current grounded results

The current grounded accepted pool is `12` systems, of which `8` remain inside
the strict per-basin crossing band and `4` are accepted only through the
relaxed crossing gate.

| System | B | Crossing mode | Crossing | Min occ | Why it matters |
|--------|---|---------------|----------|---------|----------------|
| `cal_triangle_3` | 3 | strict | 0.500 | 0.260 | Cleanest minimal polygon control |
| `cal_pentagon_5` | 5 | strict | 0.500 | 0.170 | Mid-count polygon control |
| `cal_asymmetric_3` | 3 | strict | 0.530 | 0.280 | Simple asymmetry control |
| `var_depth_gradient_4` | 4 | strict | 0.400 | 0.200 | Interpretable asymmetric-occupancy stress test |
| `var_diamond_4` | 4 | strict | 0.610 | 0.230 | Strong rotated-separatrix benchmark candidate |
| `var_l_shape_5` | 5 | strict | 0.510 | 0.120 | Non-convex geometry that stays legible |
| `hybrid_state_dep_rot_5` | 3 | strict | 0.460 | 0.220 | Hybrid mechanism that survives the exact same gate |
| `transition_routes_4` | 4 | strict | 0.430 | 0.200 | Best grounded explicit route-choice benchmark in the implemented catalog |
| `cal_hexagon_6` | 6 | relaxed | 0.540 | 0.130 | First grounded higher-basin polygon stress test |
| `cal_square_4` | 4 | relaxed | 0.330 | 0.230 | Clean 4-basin baseline with one weak basin |
| `cal_star_5` | 5 | relaxed | 0.550 | 0.160 | Radial topology variant |
| `snic_multi` | 3 | relaxed | 0.388 | 0.287 | Non-well mechanistic outlier |

Grounded basin counts currently covered by accepted systems: `3`, `4`, `5`,
and `6`. There is still no grounded accepted `7`-, `8`-, `9`-, or `10`-basin
system.

### Near-misses (fail only per-basin crossing)

- `cal_octagon_8` is still the cleanest screened high-basin retune target.
- `var_random_5a` is close but misses both occupancy and one crossing gate.
- `duffing_triple_well`, `neural_decision_3choice`, and `rational_field` are
  still scientifically interesting, but they are not currently grounded
  accepted systems.

### What failed and why

The large non-well families are not yet a validated packet. The earlier blanket
rotation fix appears to have been too aggressive for many of them: it either
collapses the endpoint structure or creates too many spurious basins. At the
moment, the grounded exceptions are limited to `snic_multi` on the relaxed side
and `hybrid_state_dep_rot_5` on the strict side.

That means the catalog should currently be read as:

- a strong rotated-multiwell control family,
- one grounded hybrid strict pass,
- one grounded non-well relaxed outlier,
- plus a real but still incomplete retune frontier.

## Active branch scope

### Paper-facing role of this catalog

Do **not** treat this catalog as the lead mechanistic story. The existing
transition-rich trio in the main branch remains the cleaner main-text causal
packet:

- `gated_local_linear` is the clean mechanistic positive
- `gated_transfer_linear` is the transfer stress test
- `multiwell_strong_transition` is a weaker secondary toy

The Claude catalog is therefore a supporting/control family for this branch,
and its active scope is now frozen to the `14` systems below.

### Active Claude-catalog subset

| System | Why it stays in scope |
|--------|------------------------|
| `arrested_spiral` | Spiral-slowdown mechanism that is not just another rotated multiwell |
| `cal_asymmetric_3` | Simple asymmetry control |
| `cal_high_cross_3` | Deliberately high-crossing control |
| `cal_hexagon_6` | Grounded relaxed-pass higher-basin polygon stress test |
| `cal_octagon_8` | High-basin near-miss kept in scope explicitly despite failing the current screen |
| `cal_pentagon_5` | Mid-count polygon control |
| `cal_square_4` | Clean square baseline with one weak-basin caveat under the strict gate |
| `checkerboard_potential` | More grid-like/alternating geometry than the polygon family |
| `duffing_triple_well` | Physically motivated triple-well near-miss worth studying directly |
| `snic_multi` | Non-well mechanistic outlier |
| `transition_routes_4` | Explicit route-choice / shared-corridor benchmark |
| `var_depth_gradient_4` | Interpretable asymmetric-occupancy stress test |
| `var_diamond_4` | Rotated-separatrix geometry mismatch |
| `var_l_shape_5` | Non-convex geometry case |

This subset intentionally mixes:

- grounded strict passes from the audit:
  `cal_asymmetric_3`, `cal_pentagon_5`, `transition_routes_4`,
  `var_depth_gradient_4`, `var_diamond_4`, `var_l_shape_5`
- grounded relaxed accepted systems:
  `cal_hexagon_6`, `cal_square_4`, `snic_multi`
- additional chosen systems outside the older `6`-system packet:
  `arrested_spiral`, `cal_high_cross_3`, `cal_octagon_8`,
  `checkerboard_potential`, `duffing_triple_well`

### What not to do

- Do not reopen full-catalog or conceptual-inventory selection for this branch.
- Do not treat the older `6`-system packet or optional second-wave language as
  the live recommendation anymore.
- Do not use the older packet tooling as the scope definition without updating
  it first.

### Historical packet tooling

The standard training stack still accepts `--env claude:<system_name>`, and the
older `6`-system packet tooling remains on disk. It should now be read as
historical scaffolding rather than the active shortlist definition:

- historical manifest:
  `skae/benchmarks/claude_catalog_packet_manifest.py`
- historical task builder:
  `tools/build_claude_catalog_packet_tasks.py`
- historical queue launcher:
  `scripts/queue_claude_catalog_packet.sh`

## Related documents

- `docs/planning/transition_rich_basin_partition_plan_20260331.md` — The original plan with acceptance gate definitions
- `docs/planning/claude_catalog_audit_20260407.md` — Independent audit by another agent
- `docs/planning/claude_catalog_senior_review_packet_20260407.md` — Senior-coauthor-facing version of the first packet without code-name-heavy framing
- `docs/planning/claude_transition_rich_catalog.md` — Our running catalog document
- `docs/planning/transition_rich_system_inventory_20260406.md` — Conceptual system inventory (from other agent)
