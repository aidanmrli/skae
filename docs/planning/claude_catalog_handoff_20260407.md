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

## What needs to be done

### Paper-facing role of this catalog

Do **not** treat this catalog as the lead mechanistic story. The existing
transition-rich trio in the main branch remains the cleaner main-text causal
packet:

- `gated_local_linear` is the clean mechanistic positive
- `gated_transfer_linear` is the transfer stress test
- `multiwell_strong_transition` is a weaker secondary toy

The Claude catalog is best used as a **benchmark-expansion and control family**
that asks whether basin-support alignment survives changes in:

- symmetry versus asymmetry
- convex versus non-convex geometry
- ordinary rotated boundaries versus explicit route reuse
- pure multiwell systems versus hybrid or non-well mechanisms
- basin count

### Recommendation: do not brute-force all registered systems

Brute-forcing LISTA and control models over the full `112`-system registry is
not a good use of compute. Most currently accepted systems share the same
independent-rotation-plus-wells mechanism, so all-systems training would add
many near-duplicate rows without adding much causal evidence.

The better move is to train on a deliberately chosen subset that spans the
distinct **basin-support alignment failure modes** we actually care about.

### Recommended subset

If we only want a minimal Claude-catalog follow-up, use these `3`:

| System | Role in the paper |
|--------|-------------------|
| `cal_triangle_3` | Minimal symmetric control |
| `var_diamond_4` | Clean geometry-mismatch / rotated-separatrix case |
| `transition_routes_4` | Explicit route-reuse benchmark |

If we want the recommended first real training packet, use these `6` strict
systems:

| System | Why it earns a training slot |
|--------|------------------------------|
| `cal_triangle_3` | Baseline: simplest grounded positive with clean geometry |
| `cal_pentagon_5` | Mid-count polygon control without adding a new mechanism confound |
| `var_depth_gradient_4` | Best simple asymmetry and occupancy-skew stress test |
| `var_diamond_4` | Best grounded oblique-boundary / rotated-separatrix system |
| `var_l_shape_5` | Best grounded non-convex geometry case |
| `transition_routes_4` | Best grounded explicit route-choice / shared-corridor system |

If we want to expand beyond `6`, stop at `9` before doing any broader sweep and
add these as the second wave:

| System | Why it is worth adding |
|--------|------------------------|
| `hybrid_state_dep_rot_5` | Strict-pass hybrid mechanism, so it tests whether the story survives a non-pure-well transport rule |
| `cal_hexagon_6` | First grounded higher-basin stress test even though it is only relaxed-pass |
| `snic_multi` | Best grounded non-well outlier if we want one clearly different mechanistic family |

### Systems to deprioritize for now

- `cal_square_4`: too redundant once `var_diamond_4` and `transition_routes_4`
  are in the packet
- `cal_star_5`: interesting, but less causally clean than the L-shape and route
  systems
- `cal_asymmetric_3`: useful, but lower leverage once `var_depth_gradient_4`
  already covers asymmetry in a richer way
- random or boundary-tuned variants such as `cal_high_cross_3`, `cal_low_cross_4`,
  `var_random_*`, and similar parameter-perturbation controls: these are useful
  later for robustness, not for the first coauthor-facing packet

### If we spend more screening or retuning budget

Only a few next steps look worth it:

1. Retune `cal_octagon_8`, because it is the cleanest remaining screened
   high-basin target.
2. Screen or refresh only the unscreened systems that add new mechanism value,
   especially `non_voronoi_basins` and `hybrid_rotating_centers_3`.
3. Do **not** spend early budget on more near-duplicate polygon or random-layout
   variants.

The practical recommendation is therefore:

- main mechanistic paper story: existing calibrated trio
- Claude-catalog training packet: `6` strict systems above
- optional extension packet: add the `3` second-wave systems above
- extra retune work only if we specifically need a stronger `6+`-basin control

### Immediate training packet

If we actually launch runs now, the first packet should stay small and use the
same paper-facing model family we already trust elsewhere:

- systems:
  `claude:cal_triangle_3`,
  `claude:cal_pentagon_5`,
  `claude:var_depth_gradient_4`,
  `claude:var_diamond_4`,
  `claude:var_l_shape_5`,
  `claude:transition_routes_4`
- seeds: `0,1,2`
- sequence length: `8`
- target size: `256`
- budget: `200k` steps
- models:
  `generic_sparse_ns200k_best`,
  `generic_sparse_sc0_ns200k_best`,
  `lista_dense_promoted_stage4`

Use the following recipe mapping when converting those paper labels into CLI
arguments:

| Paper label | Base config / key overrides |
| --- | --- |
| `generic_sparse_ns200k_best` | `--config generic_sparse --num_steps 200000 --lr 1e-4 --k_matrix_lr 1e-5 --weight_decay 1e-4 --reconst_coeff 0.03 --pred_coeff 1.0 --sparsity_coeff 0.0025 --target_size 256 --sequence_length 8` |
| `generic_sparse_sc0_ns200k_best` | same as `generic_sparse_ns200k_best`, but `--sparsity_coeff 0.0` |
| `lista_dense_promoted_stage4` | `--config lista_parity_generic_sparse --num_steps 200000 --lr 5e-5 --k_matrix_lr 5e-6 --weight_decay 1e-4 --reconst_coeff 0.03 --pred_coeff 1.0 --sparsity_coeff 0.003 --target_size 256 --sequence_length 8 --lista_alpha 0.15 --lista_num_loops 1 --lista_final_op relu --k_structure dense` |

Example command template:

Run the actual command only inside a compute allocation, not on the login
node.

```bash
uv run python tools/train.py \
  --config generic_sparse \
  --env claude:cal_triangle_3 \
  --num_steps 200000 \
  --target_size 256 \
  --sequence_length 8 \
  --lr 1e-4 \
  --k_matrix_lr 1e-5 \
  --weight_decay 1e-4 \
  --reconst_coeff 0.03 \
  --pred_coeff 1.0 \
  --sparsity_coeff 0.0025 \
  --seed 0
```

Recommended full-packet launch:

Submit the queue launcher itself through SLURM so the task-building Python
steps also run on a compute node:

```bash
sbatch scripts/queue_claude_catalog_packet.sh
```

Do not broaden beyond that `6 x 3 x 3` matrix until we know whether the
catalog actually adds something beyond the existing transition-rich trio.

Runnable packet files:

- manifest:
  `skae/benchmarks/claude_catalog_packet_manifest.py`
- task builder:
  `tools/build_claude_catalog_packet_tasks.py`
- queue launcher:
  `scripts/queue_claude_catalog_packet.sh`

## Related documents

- `docs/planning/transition_rich_basin_partition_plan_20260331.md` — The original plan with acceptance gate definitions
- `docs/planning/claude_catalog_audit_20260407.md` — Independent audit by another agent
- `docs/planning/claude_catalog_senior_review_packet_20260407.md` — Senior-coauthor-facing version of the first packet without code-name-heavy framing
- `docs/planning/claude_transition_rich_catalog.md` — Our running catalog document
- `docs/planning/transition_rich_system_inventory_20260406.md` — Conceptual system inventory (from other agent)
