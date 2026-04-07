# Transition-Rich System Catalog — Handoff

Date: 2026-04-07  
Branch: `claude-transition-rich-systems-gen`  
Prior conversation produced 9 commits on this branch.

## What was built

A catalog of **117 2D autonomous dynamical systems** in `skae/claude_catalog/` for
benchmarking basin-aware Koopman autoencoders. Each system is a class inheriting
from `CatalogSystem` (defined in `skae/claude_catalog/base.py`) with `dynamics()`,
`reset()`, `step()` methods using RK4 integration on `torch.float64` tensors. A
decorator-based registry (`skae/claude_catalog/registry.py`) maps string names to
classes.

### Module layout

| Module | Count | Contents |
|--------|-------|---------|
| `systems_gradient.py` | 17 | Gaussian-well gradient flows + proportional rotation (Categories A/B) |
| `systems_bio_physical.py` | 17 | Toggle switch, Duffing, Josephson, cusp catastrophe, etc. (C/D/F) |
| `systems_creative.py` | 17 | SNIC, heteroclinic cycle, maze, multi-scroll, Brusselator, etc. (E/G/H) |
| `systems_novel.py` | 17 | Waddington landscape, protein folding, climate tipping, neural decision, etc. (H) |
| `systems_tuned.py` | 21 | Calibrated Gaussian wells with independent rotation — designed to pass gates (B/H) |
| `systems_variants.py` | 17 | Diverse well geometries: L-shape, diamond, zigzag, depth gradient, etc. (B/H) |
| `systems_hybrid.py` | 6 | Well templates + creative dynamics: anisotropic, limit cycle, saddle bridge (H) |
| `systems_flagship.py` | 5 | Paper-motivated: Voronoi mismatch, frequency discrimination, damping gradient (H) |

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

**17 systems pass all strict acceptance gates** via `fast_screen_catalog.py`:

| System | B | Cross | Occ | Source |
|--------|---|-------|-----|--------|
| `cal_triangle_3` | 3 | 50% | 0.26 | `systems_tuned.py` |
| `cal_square_4` | 4 | 33% | 0.23 | `systems_tuned.py` |
| `cal_pentagon_5` | 5 | 50% | 0.17 | `systems_tuned.py` (retuned ω=1.1) |
| `cal_asymmetric_3` | 3 | 53% | 0.28 | `systems_tuned.py` |
| `cal_star_5` | 5 | 55% | 0.16 | `systems_tuned.py` |
| `cal_high_cross_3` | 3 | 61% | 0.28 | `systems_tuned.py` |
| `cal_low_cross_4` | 4 | 42% | 0.22 | `systems_tuned.py` |
| `var_diamond_4` | 4 | 61% | 0.23 | `systems_variants.py` |
| `var_l_shape_5` | 5 | 51% | 0.12 | `systems_variants.py` |
| `var_depth_gradient_4` | 4 | 40% | 0.20 | `systems_variants.py` (retuned) |
| `var_mixed_widths_5` | 5 | 42% | 0.15 | `systems_variants.py` |
| `var_grid_2x2` | 4 | 38% | 0.23 | `systems_variants.py` |
| `var_random_4a` | 4 | 55% | 0.20 | `systems_variants.py` |
| `hybrid_state_dep_rot_5` | 3 | 46% | 0.22 | `systems_hybrid.py` |
| `hybrid_rotating_centers_3` | 3 | 47% | 0.27 | `systems_hybrid.py` |
| `transition_routes_4` | 4 | 43% | 0.20 | `systems_tuned.py` |
| `snic_multi` | 3 | 39% | 0.29 | `systems_creative.py` (only non-well pass) |

Basin counts covered: 3, 4, 5. Missing: 6, 7, 8, 9, 10.

### Near-misses (fail only per-basin crossing)

- `cal_hexagon_6` (6B): one basin at 0.22 crossing, rest fine. ω=0.90.
- `cal_octagon_8` (8B): three basins over 0.70 crossing. ω=0.90.
- `flagship_voronoi_mismatch` (4B): 40% overall but per-basin spread too wide.
- `var_depth_gradient_4` before retune, `duffing_triple_well`: one basin over-transitions.

### What failed and why

**Non-well systems (bio/physical/creative/novel)**: A blanket rotation fix added
`self.omega = 0.8–1.0` and `dxdt += omega*y; dydt -= omega*x` to all 51 non-well
systems. This was far too aggressive — it either merged basins into 1 (rotation
overwhelmed existing dynamics) or created 15+ spurious basins with 100% crossing.

These systems need **per-system omega tuning** at much lower values (0.1–0.3), or
entirely different transition mechanisms. The `snic_multi` system is the only non-well
system that passes, because it has natural multistability from its phase dynamics.

## What needs to be done

### High priority — push basin count range to 6+

1. **Fix `cal_hexagon_6`**: One basin's crossing is 0.22. The windward/leeward
   asymmetry from uniform rotation is the root cause. Potential fixes:
   - Non-uniform well depths (make leeward wells deeper)
   - Position-dependent omega (stronger rotation near low-crossing basins)
   - Accept the relaxed crossing check (overall-only, not per-basin)

2. **Fix `cal_octagon_8`**: Three basins over 0.70 crossing (0.73–0.79). Same
   windward/leeward issue. Try ω=0.85 or non-uniform well depths.

3. **Try odd-count polygons at 7 and 9**: The heptagon (7B) and nonagon (9B)
   should be less susceptible to the even-polygon windward/leeward issue. These
   exist as numpy-validated configs but not yet as registered classes or
   `fast_screen_catalog.py`-validated entries.

### Medium priority — non-well mechanistic diversity

4. **Revert and retune non-well systems**: The blanket omega=0.8–1.0 fix in
   `systems_bio_physical.py`, `systems_creative.py`, `systems_novel.py` broke them.
   For each system:
   - Revert to omega=0 (original dynamics)
   - Check if it passes without rotation (some like `duffing_triple_well` had 66% crossing before the fix)
   - If not, add omega=0.1–0.3 and rescreen
   - Priority targets: `duffing_triple_well` (3B, 66% pre-fix), `arrested_spiral` (5B, 49% pre-fix), `protein_folding_landscape` (3B, 60% pre-fix)

5. **Screen the blended-dynamics flagships**: `flagship_freq_discrimination`,
   `flagship_damping_gradient`, `flagship_rotation_reversal` use torch softmax
   blending which makes vmap-based screening extremely slow (~20 min per system).
   Either:
   - Run on a compute node with `salloc` (never on login node!)
   - Rewrite their dynamics to be vmap-compatible
   - Write numpy equivalents for screening

### Low priority — integration and polish

6. **Integrate passing systems into `skae/data.py` registry** so models can train on them.
   Each needs a config dataclass in `skae/config.py` and an entry in `_ENV_REGISTRY`.

7. **Generate true basin-of-attraction maps** for all 17 passing systems. Currently done
   for ~9 systems. Use the numpy-based approach in `tools/plot_catalog_gallery.py` (120×120
   grid, ~4 min per system).

8. **Write tests**: Determinism test, crossing metric test, registry smoke test. These
   go in `tests/test_fast_screen_catalog.py` or similar.

## Important warnings

- **NEVER run programs on the login node.** See Compute Node Policy in CLAUDE.md.
  All `uv run python ...` and `pytest` invocations require `salloc` first.
- The `fast_screen_catalog.py` imports are fragile — it must import ALL system modules
  to populate the registry. If you add a new module, add its import there.
- The screening is slow: ~2 min per system with vmap, ~15–20 min for blended-dynamics
  systems. Plan accordingly.
- Saved screening artifacts are in `results/` which is gitignored. The audit document
  at `docs/planning/claude_catalog_audit_20260407.md` reconciles what's actually saved.

## Related documents

- `docs/planning/transition_rich_basin_partition_plan_20260331.md` — The original plan with acceptance gate definitions
- `docs/planning/claude_catalog_audit_20260407.md` — Independent audit by another agent
- `docs/planning/claude_transition_rich_catalog.md` — Our running catalog document
- `docs/planning/transition_rich_system_inventory_20260406.md` — Conceptual system inventory (from other agent)
