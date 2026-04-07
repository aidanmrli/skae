# Claude: Transition-Rich System Catalog

Branch: `claude-transition-rich-systems-gen`  
Generated: 2026-04-06, updated 2026-04-07

## Summary

- **106 systems** implemented across 6 modules in `skae/claude_catalog/`
- **18 systems confirmed passing** all acceptance gates (numpy validation)
- **~25-30 systems estimated passing** after rotation fix
- **10-20 systems selected** for benchmark (see below)

## Key Calibration Result

**Independent rotation** `dx/dt += ω·y, dy/dt -= ω·x` is the mechanism that
produces transition-rich dynamics. The crossing fraction is controlled by:

```
crossing ≈ f(ω / amp)    where ω/amp ∈ [0.3, 0.7] → crossing ∈ [0.30, 0.65]
```

- **Proportional rotation** (α·R·∇V): gives < 15% crossing — insufficient
- **Independent rotation**: adds constant angular velocity, creating spiral
  approach to basins that carries trajectories across boundaries

### Calibrated parameter table (Gaussian wells on polygon, σ=0.5, radius=1.8)

| Basins | Well amp | Omega | Crossing | Confinement |
|--------|----------|-------|----------|-------------|
| 3 | 2.0 | 1.0 | 47% | 0.03 |
| 4 | 3.0 | 1.0 | 45% | 0.03 |
| 5 | 2.0 | 1.0 | 41% | 0.03 |
| 6 | 2.0 | 0.8 | 44% | 0.03 |
| 8 | 3.0 | 1.2 | 55% | 0.02 |

## Confirmed Passing Systems (numpy-validated)

| # | System | Basins | Crossing | Min Occ | Type |
|---|--------|--------|----------|---------|------|
| 1 | `cal_triangle_3` | 3 | 48% | 0.26 | Polygon (equilateral) |
| 2 | `cal_square_4` | 4 | 45% | 0.23 | Polygon (square) |
| 3 | `cal_pentagon_5` | 5 | 41% | 0.17 | Polygon (pentagon) |
| 4 | `cal_hexagon_6` | 6 | 44% | 0.10 | Polygon (hexagon) |
| 5 | `cal_octagon_8` | 8 | 65% | 0.06 | Polygon (octagon) |
| 6 | `cal_asymmetric_3` | 3 | 50% | 0.25 | Asymmetric depths |
| 7 | `cal_star_5` | 5 | 46% | 0.16 | Star topology |
| 8 | `cal_high_cross_3` | 3 | 62% | 0.26 | High rotation |
| 9 | `cal_low_cross_4` | 4 | 45% | 0.23 | Moderate rotation |
| 10 | `var_random_3a` | 3 | 51% | 0.24 | Random placement |
| 11 | `var_random_4a` | 4 | 51% | 0.21 | Random placement |
| 12 | `var_random_5a` | 5 | 50% | 0.14 | Random placement |
| 13 | `var_depth_gradient_4` | 4 | 41% | 0.20 | Depth gradient |
| 14 | `var_mixed_widths_5` | 5 | 41% | 0.17 | Mixed widths |
| 15 | `var_mild_rot_5` | 5 | 30% | 0.17 | Minimal crossing |
| 16 | `var_diamond_4` | 4 | 54% | 0.20 | Rotated square |
| 17 | `var_l_shape_5` | 5 | 40% | 0.11 | L-shape topology |
| 18 | `var_grid_2x2` | 4 | 33% | 0.21 | Grid layout |

### Likely passing after rotation fix (needs full torch validation)

| System | Pre-fix basins | Pre-fix cross | Expected post-fix |
|--------|---------------|---------------|-------------------|
| `duffing_triple_well` | 3 | 66% | ~50% (already high) |
| `competitive_exclusion_3` | 3 | 25% | ~40% |
| `dna_regulatory_switch` | 4 | 20% | ~35% |
| `neural_decision_3choice` | 6 | 23% | ~38% |
| `alluvial_fan` | 5 | 16% | ~31% |
| `mixed_dynamics_triple` | 3 | ~15% | ~35% |
| `non_voronoi_basins` | 3 | ~15% | ~35% |
| `slow_fast_triple` | 3 | ~10% | ~30% |
| `muller_brown_rotated` | 3 | ~12% | ~30% |
| `transition_routes_4` | 4 | ~10% | ~30% |

## Benchmark Selection (15 systems)

Selected for diversity across: basin count (3-8), crossing range (30-65%),
topology (symmetric/asymmetric/non-convex/hierarchical), and dynamics type.

### Tier 1: Core benchmark (8 systems)

These are the most important systems for the paper.

| # | System | B | C | Why included |
|---|--------|---|---|--------------|
| 1 | `cal_triangle_3` | 3 | 48% | Clean 3-basin baseline |
| 2 | `cal_pentagon_5` | 5 | 41% | Mid-complexity, symmetric |
| 3 | `cal_octagon_8` | 8 | 65% | Scalability to many basins |
| 4 | `cal_asymmetric_3` | 3 | 50% | Tests robustness to depth imbalance |
| 5 | `var_depth_gradient_4` | 4 | 41% | Different basin depths |
| 6 | `var_l_shape_5` | 5 | 40% | Non-convex topology |
| 7 | `cal_star_5` | 5 | 46% | Star routing topology |
| 8 | `var_diamond_4` | 4 | 54% | Rotated geometry |

### Tier 2: Stretch systems (7 systems)

These add diversity but need more validation.

| # | System | B | C | Why included |
|---|--------|---|---|--------------|
| 9 | `cal_high_cross_3` | 3 | 62% | High crossing endpoint |
| 10 | `var_mild_rot_5` | 5 | 30% | Low crossing endpoint |
| 11 | `mixed_dynamics_triple` | 3 | ~35% | Different dynamics per basin |
| 12 | `non_voronoi_basins` | 3 | ~35% | Non-Voronoi boundaries |
| 13 | `duffing_triple_well` | 3 | ~50% | Classic physical system |
| 14 | `muller_brown_rotated` | 3 | ~30% | Computational chemistry |
| 15 | `transition_routes_4` | 4 | ~30% | Route-specific transitions |

## Module Organization

| Module | Count | Category |
|--------|-------|----------|
| `systems_gradient.py` | 17 | A/B: gradient + proportional rotation |
| `systems_bio_physical.py` | 17 | C/D: biological + physical |
| `systems_creative.py` | 17 | E/F/G/H: creative + abstract |
| `systems_novel.py` | 17 | H: novel real-world-inspired |
| `systems_tuned.py` | 21 | B/H: calibrated to pass gates |
| `systems_variants.py` | 17 | B/H: diverse well configurations |
| **Total** | **106** | |

## Visualizations

- Phase portraits: `results/claude_catalog_validation/plots/`
- Gallery composite: `results/claude_catalog_validation/gallery/system_gallery.png`
- Crossing comparison: `results/claude_catalog_validation/gallery/crossing_comparison.png`

## Failure Analysis

| Failure mode | Count | Root cause | Fix |
|---|---|---|---|
| Low crossing (<30%) | ~40 | No independent rotation | Add ω·y, -ω·x |
| Too many basins (>10) | ~8 | Clustering too sensitive | Increase cluster tolerance |
| Too few basins (<3) | ~10 | Wrong parameter regime | Redesign dynamics |
| Low occupancy (<5%) | ~12 | Rotation creates asymmetry | Balance wells |
| High crossing (>70%) | ~5 | Too much rotation/chaos | Reduce omega |
