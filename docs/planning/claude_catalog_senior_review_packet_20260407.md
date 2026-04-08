# Claude Catalog: Senior-Review Follow-Up Packet

Date: 2026-04-07

This note is the senior-coauthor-facing version of the Claude-catalog follow-up
recommendation. It is written to make the experimental choice and causal
comparison clear without depending on internal system code names in the main
protocol description.

## Decision to make

The question is not whether the implemented catalog is large. The question is
whether a **small, grounded benchmark-expansion packet** should be added to the
paper to test whether basin-support alignment survives beyond the existing
three-system mechanistic toy suite.

The recommended answer is:

- keep the existing three-system transition-rich suite as the lead mechanistic
  story
- use a small Claude-catalog packet only as a benchmark-expansion and control
  family
- start with a deliberately chosen `6`-system packet rather than a broad sweep

## Current grounded state

The implemented catalog is real, but still selective rather than broad:

- `112` systems are registered in the current worktree
- `83` systems have grounded fast-screen coverage
- `12` systems currently pass the official acceptance rule
- `8` of those `12` remain inside the strict per-basin crossing band

That is enough to justify a **small follow-up packet**. It is not enough to
justify writing the catalog as a validated large benchmark family.

## Recommended first packet

The first packet should contain `6` systems chosen to span the main structural
failure modes we care about.

| Descriptive label | Basin count | What it isolates |
| --- | --- | --- |
| Three-basin symmetric triangle | `3` | Minimal clean control with visible symmetry |
| Five-basin symmetric pentagon | `5` | More basins without changing the core mechanism |
| Four-basin depth-graded layout | `4` | Simple asymmetry and occupancy skew |
| Four-basin rotated diamond | `4` | Boundary geometry that departs from the visible nearest-center picture |
| Five-basin L-shaped layout | `5` | Non-convex basin geometry |
| Four-basin shared-corridor routes | `4` | Explicit route reuse and bottleneck structure |

This packet is useful because it asks whether the same learned representation
continues to work when we change:

- basin count
- symmetry versus asymmetry
- convex versus non-convex geometry
- ordinary rotated boundaries versus explicit route reuse

## Exact causal protocol

The intended comparison is a strict matched-family packet, not a broad search.

Compare the following three model families on all `6` systems:

- **Sparse MLP anchor**
- **Zero-sparsity MLP control**
- **Dense LISTA comparator**

Hold the following fixed across the whole packet:

- training budget: `200k` optimization steps
- latent width: `256`
- training horizon: `H = 8`
- batch size: `256`
- seeds: `0, 1, 2`
- system-specific step size: use the default stable step size already assigned
  to each benchmark system
- checkpoint rule and forecasting evaluation: keep exactly the same paper-facing
  training and evaluation protocol used in the existing `200k` packet

The causal comparisons are:

- **Sparse MLP anchor vs zero-sparsity MLP control**:
  does explicit sparsity improve reusable basin-support structure beyond the
  same MLP architecture without the sparsity penalty?
- **Sparse MLP anchor vs dense LISTA comparator**:
  does the LISTA-style encoder help on these transition-rich control systems
  under the same budget and same front-end width?
- **Pattern across the six systems**:
  are the gains robust across symmetry changes, geometric mismatch, non-convex
  support, and explicit route reuse?

## How to interpret outcomes

- If the sparse MLP anchor beats the zero-sparsity control on the rotated,
  non-convex, or shared-corridor systems, that strengthens the claim that
  explicit sparsity helps recover reusable local partitions.
- If the sparse and zero-sparsity MLPs tie while both outperform dense LISTA,
  then the packet is still useful as a benchmark/control family, but it does
  not strengthen a sparse-specific mechanism claim.
- If all models degrade mainly on the rotated-diamond, L-shaped, and
  shared-corridor systems, those systems become useful hard controls even if
  they do not become new main-text positives.
- If only the symmetric polygon controls are strong positives, the catalog
  should stay in a supporting/control role rather than become a major paper
  family.

## What not to do yet

- Do not run the full registered catalog.
- Do not spend the first follow-up budget on near-duplicate polygon variants or
  random-layout perturbations.
- Do not expand to higher-basin systems before we know whether the `6`-system
  packet adds meaningful paper signal.

## Optional second wave

Only expand beyond the first `6` systems if the first packet clearly adds paper
value. The next three systems to add are:

- a strict-pass hybrid transport system
- a relaxed-pass six-basin polygon stress test
- a relaxed-pass non-well multistable outlier

These add mechanistic breadth, but they are not the first thing to run.

## Internal mapping for implementers

The descriptive packet above maps to the following internal environment names:

| Descriptive label | Internal environment |
| --- | --- |
| Three-basin symmetric triangle | `claude:cal_triangle_3` |
| Five-basin symmetric pentagon | `claude:cal_pentagon_5` |
| Four-basin depth-graded layout | `claude:var_depth_gradient_4` |
| Four-basin rotated diamond | `claude:var_diamond_4` |
| Five-basin L-shaped layout | `claude:var_l_shape_5` |
| Four-basin shared-corridor routes | `claude:transition_routes_4` |

The model-family mapping is:

| Senior-review name | Internal variant |
| --- | --- |
| Sparse MLP anchor | `generic_sparse_ns200k_best` |
| Zero-sparsity MLP control | `generic_sparse_sc0_ns200k_best` |
| Dense LISTA comparator | `lista_dense_promoted_stage4` |

The implementation-side queue path is:

- packet handoff:
  [claude_catalog_handoff_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_handoff_20260407.md)
- packet manifest:
  [claude_catalog_packet_manifest.py](/home/mila/l/lia/skae/skae/benchmarks/claude_catalog_packet_manifest.py)
- task builder:
  [build_claude_catalog_packet_tasks.py](/home/mila/l/lia/skae/tools/build_claude_catalog_packet_tasks.py)
- SLURM launcher:
  [queue_claude_catalog_packet.sh](/home/mila/l/lia/skae/scripts/queue_claude_catalog_packet.sh)
