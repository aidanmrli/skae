# Claude Catalog: Senior-Review Scope Note

Date: 2026-04-08

This note records the senior-coauthor-facing scope for the Claude-catalog part
of the branch. It replaces the older `6`-system follow-up recommendation as the
current scope note.

## Active branch scope

The overall branch experiment shortlist is now fixed to `17` systems:

- existing native transition-rich trio:
  `multiwell_strong_transition`, `gated_local_linear`,
  `gated_transfer_linear`
- Claude-catalog subset:
  `arrested_spiral`, `cal_asymmetric_3`, `cal_high_cross_3`,
  `cal_hexagon_6`, `cal_octagon_8`, `cal_pentagon_5`, `cal_square_4`,
  `checkerboard_potential`, `duffing_triple_well`, `snic_multi`,
  `transition_routes_4`, `var_depth_gradient_4`, `var_diamond_4`,
  `var_l_shape_5`

This note concerns the `14` Claude-catalog members of that fixed shortlist.

## Why this is the right Claude subset

The implemented catalog is real, but still selective rather than broad:

- `112` systems are registered in the current worktree
- `83` systems have grounded fast-screen coverage
- `12` systems currently pass the official acceptance rule
- `8` of those `12` remain inside the strict per-basin crossing band

That grounding is useful, but it no longer defines the active packet by
itself. The active Claude subset is chosen to mix:

- grounded controls that already clear the current screen
- one grounded non-well outlier
- several additional systems we explicitly want to study even though they were
  not part of the older recommended `6`-system packet

## Active Claude-catalog systems

| Descriptive label | Internal environment | What it isolates |
| --- | --- | --- |
| Arrested spiral | `claude:arrested_spiral` | Spiral-slowdown transport rather than simple rotated wells |
| Three-basin asymmetric control | `claude:cal_asymmetric_3` | Simple asymmetry |
| Three-basin high-crossing control | `claude:cal_high_cross_3` | High-crossing control regime |
| Six-basin symmetric hexagon | `claude:cal_hexagon_6` | Higher-basin polygon stress test that currently clears only the relaxed gate |
| Eight-basin symmetric octagon | `claude:cal_octagon_8` | High-basin near miss kept in scope explicitly despite failing the current screen |
| Five-basin symmetric pentagon | `claude:cal_pentagon_5` | Mid-count polygon control |
| Four-basin symmetric square | `claude:cal_square_4` | Clean square baseline with one strict-gate caveat |
| Checkerboard potential | `claude:checkerboard_potential` | Alternating/grid-like geometry |
| Duffing triple well | `claude:duffing_triple_well` | Physically motivated triple-well mechanism |
| SNIC multistable system | `claude:snic_multi` | Non-well mechanistic outlier |
| Four-basin shared-corridor routes | `claude:transition_routes_4` | Explicit route reuse and bottleneck structure |
| Four-basin depth-graded layout | `claude:var_depth_gradient_4` | Asymmetry and occupancy skew |
| Four-basin rotated diamond | `claude:var_diamond_4` | Rotated-separatrix geometry mismatch |
| Five-basin L-shaped layout | `claude:var_l_shape_5` | Non-convex basin geometry |

## Causal protocol

For any forward Claude-catalog experiment on this branch, keep the same matched
three-family comparison already used elsewhere:

- **Sparse MLP anchor**
- **Zero-sparsity MLP control**
- **Dense LISTA comparator**

Hold fixed the usual diagnostic recipe unless a document explicitly says
otherwise:

- training budget: `20k` optimization steps while interpretability metrics and
  protocol details are still being set; reserve `200k` only for the final
  paper-facing rerun once the branch recipe is locked
- latent width: `256`
- training horizon: `H = 8`
- batch size: `256`
- seeds: `0, 1, 2`
- system-specific step size: the default stable step size for each system

The causal question is no longer “which Claude systems should we study?” The
question is how these fixed chosen systems differ in symmetry, geometry,
crossing regime, and mechanism, and whether those differences change the
representation story.

## What not to do

- Do not run the full registered catalog.
- Do not treat the older `6`-system packet or optional second-wave language as
  the current plan.
- Do not use the existing packet manifest/task-builder/launcher as the scope
  definition without updating them first; they record the superseded `6`-system
  packet.

## Historical tooling note

The older packet scaffolding is still on disk, but it is now historical
infrastructure rather than the active scope definition:

- packet handoff:
  [claude_catalog_handoff_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_handoff_20260407.md)
- historical packet manifest:
  [claude_catalog_packet_manifest.py](/home/mila/l/lia/skae/skae/benchmarks/claude_catalog_packet_manifest.py)
- historical task builder:
  [build_claude_catalog_packet_tasks.py](/home/mila/l/lia/skae/tools/build_claude_catalog_packet_tasks.py)
- historical SLURM launcher:
  [queue_claude_catalog_packet.sh](/home/mila/l/lia/skae/scripts/queue_claude_catalog_packet.sh)
