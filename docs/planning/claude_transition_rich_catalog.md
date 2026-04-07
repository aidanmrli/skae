# Claude: Transition-Rich System Catalog

Branch: `claude-transition-rich-systems-gen`  
Generated: 2026-04-06, updated 2026-04-07

## Audit Status

This note is now partially stale and should not be used as the source of truth
for benchmark readiness.

The current artifact-backed audit is:
[claude_catalog_audit_20260407.md](/home/mila/l/lia/skae/docs/planning/claude_catalog_audit_20260407.md).

That audit shows:

- `112` registered systems,
- `83` systems now covered by the combined grounded screen,
- `29` implemented but still unscreened systems,
- `12` accepted systems in the currently grounded screen,
- and an `8`-system strict-crossing core inside that accepted pool.

So the `44 confirmed passing` headline below is preserved as historical branch
provenance, not as the current validated result.

Post-retune screening (this branch) confirmed 6 additional passes beyond the
audit's 12: `cal_pentagon_5`, `var_depth_gradient_4`, `var_mixed_widths_5`,
`var_grid_2x2`, `var_random_4a`, `cal_high_cross_3`, `cal_low_cross_4`,
`hybrid_rotating_centers_3`. Non-well systems all fail after rotation fix
(omega too aggressive; needs 0.1-0.3 not 0.8-1.0 for existing dynamics).

## Summary

- **118 systems** implemented across 8 modules in `skae/claude_catalog/`
- **17 systems confirmed passing** strict fast-screen gates (auditable artifacts)
- **~8 near-misses** that fail only per-basin crossing (hexagon, octagon, etc.)
- Non-well systems need system-specific omega tuning (current rotation fix broke them)

## Key Calibration Finding

**Independent rotation** `dx/dt += ω·y, dy/dt -= ω·x` produces transition-rich dynamics.

```
crossing_fraction ≈ f(ω / amplitude)
  ω/amp ∈ [0.3, 0.7] → crossing ∈ [30%, 65%]
```

| Basins | Amplitude | Omega | Crossing | Notes |
|--------|-----------|-------|----------|-------|
| 3 | 2.0 | 1.0 | 47% | Sweet spot |
| 4 | 3.0 | 1.0 | 45% | Robust |
| 5 | 2.0 | 1.0 | 41% | Robust |
| 6 | 2.0 | 0.8 | 44% | Lower omega needed |
| 7 | 2.5 | 0.9 | 51% | |
| 8 | 3.0 | 1.2 | 55% | Higher omega needed |
| 9 | 3.0 | 1.0 | 69% | Near upper bound |
| 10 | 3.5 | 1.0 | 64% | |

## All 44 Confirmed Passing Systems

### Calibrated polygon (9 systems)

| Name | B | C% | Occ | Geometry |
|------|---|----|----|----------|
| `cal_triangle_3` | 3 | 48 | 0.26 | Equilateral triangle |
| `cal_square_4` | 4 | 45 | 0.23 | Square |
| `cal_pentagon_5` | 5 | 41 | 0.17 | Pentagon |
| `cal_hexagon_6` | 6 | 44 | 0.10 | Hexagon |
| `cal_octagon_8` | 8 | 65 | 0.06 | Octagon |
| `cal_asymmetric_3` | 3 | 50 | 0.25 | Unequal depths |
| `cal_star_5` | 5 | 46 | 0.16 | Star topology |
| `cal_high_cross_3` | 3 | 62 | 0.26 | High rotation |
| `cal_low_cross_4` | 4 | 45 | 0.23 | Moderate rotation |

### Variant well configs (9 systems)

| Name | B | C% | Occ | Geometry |
|------|---|----|----|----------|
| `var_random_3a` | 3 | 51 | 0.24 | Random |
| `var_random_4a` | 4 | 51 | 0.21 | Random |
| `var_random_5a` | 5 | 50 | 0.14 | Random |
| `var_depth_gradient_4` | 4 | 41 | 0.20 | Depth cascade |
| `var_mixed_widths_5` | 5 | 41 | 0.17 | Mixed σ |
| `var_mild_rot_5` | 5 | 30 | 0.17 | Minimal crossing |
| `var_diamond_4` | 4 | 54 | 0.20 | Rotated square |
| `var_l_shape_5` | 5 | 40 | 0.11 | L-shape |
| `var_grid_2x2` | 4 | 33 | 0.21 | Grid |

### Diverse geometries batch 1 (13 systems)

| Name | B | C% | Occ | Geometry |
|------|---|----|----|----------|
| `tri_rot0` | 3 | 39 | 0.29 | Triangle (0° offset) |
| `tri_rot29` | 3 | 46 | 0.30 | Triangle (30°) |
| `tri_rot59` | 3 | 45 | 0.28 | Triangle (60°) |
| `V_shape_4` | 4 | 36 | 0.12 | V-shape |
| `bow_tie_4` | 4 | 45 | 0.21 | Bow-tie |
| `narrow_4` | 4 | 47 | 0.24 | Narrow wells |
| `wide_4` | 4 | 36 | 0.20 | Wide wells |
| `pent_w9` | 5 | 31 | 0.16 | Pentagon (ω=0.9) |
| `pent_w11` | 5 | 40 | 0.19 | Pentagon (ω=1.1) |
| `pent_w13` | 5 | 46 | 0.16 | Pentagon (ω=1.3) |
| `pent_w15` | 5 | 49 | 0.17 | Pentagon (ω=1.5) |
| `heptagon_7` | 7 | 51 | 0.07 | 7-well heptagon |
| `decagon_10` | 10 | 64 | 0.05 | 10-well decagon |

### Diverse geometries batch 2 (13 systems)

| Name | B | C% | Occ | Geometry |
|------|---|----|----|----------|
| `H_shape_5` | 5 | 34 | 0.10 | H-shape |
| `hex_deep_6` | 6 | 36 | 0.10 | Deep hexagon |
| `hex_wide_6` | 6 | 46 | 0.11 | Wide hexagon |
| `irreg_3a` | 3 | 41 | 0.20 | Irregular triangle |
| `irreg_3b` | 3 | 47 | 0.24 | Irregular triangle |
| `irreg_4a` | 4 | 50 | 0.21 | Irregular quad |
| `irreg_4b` | 4 | 51 | 0.17 | Irregular quad |
| `pent_r15` | 5 | 30 | 0.17 | Small pentagon |
| `pent_r22` | 5 | 46 | 0.16 | Large pentagon |
| `spiral_arm_5` | 4 | 50 | 0.16 | Spiral arms |
| `sq_large_4` | 4 | 44 | 0.20 | Large square |
| `sq_mixed_4` | 4 | 40 | 0.24 | Mixed depths |
| `sq_tight_4` | 4 | 40 | 0.24 | Tight square |

## Benchmark Selection (15 systems)

Selected for maximum diversity across: basin count (3–10), crossing range
(30–65%), topology (polygon/non-convex/hierarchical), well balance, and
unique properties.

| # | Name | B | C% | Selection rationale |
|---|------|---|----|----|
| 1 | `cal_triangle_3` | 3 | 48 | **Baseline**: simplest multi-basin system |
| 2 | `cal_pentagon_5` | 5 | 41 | **Mid-complexity**: standard polygon |
| 3 | `cal_octagon_8` | 8 | 65 | **Scalability**: many basins, high crossing |
| 4 | `cal_asymmetric_3` | 3 | 50 | **Robustness**: unequal well depths |
| 5 | `var_depth_gradient_4` | 4 | 41 | **Asymmetry**: cascading well depths |
| 6 | `var_l_shape_5` | 5 | 40 | **Non-convex**: L-shaped topology |
| 7 | `cal_star_5` | 5 | 46 | **Star**: radiating topology |
| 8 | `heptagon_7` | 7 | 51 | **7-basin**: odd-count basins |
| 9 | `decagon_10` | 10 | 64 | **High-count**: 10 basins, scalability test |
| 10 | `var_mild_rot_5` | 5 | 30 | **Low crossing**: boundary condition |
| 11 | `cal_high_cross_3` | 3 | 62 | **High crossing**: boundary condition |
| 12 | `bow_tie_4` | 4 | 45 | **Non-standard**: bow-tie geometry |
| 13 | `spiral_arm_5` | 4 | 50 | **Non-trivial**: spiral arm placement |
| 14 | `H_shape_5` | 5 | 34 | **Topology**: H-shape connectivity |
| 15 | `hex_wide_6` | 6 | 46 | **Width variation**: wide basins |

### Key properties of the benchmark

- **Basin count range**: 3–10 (6 different counts)
- **Crossing range**: 30%–65%
- **Topologies**: polygon, star, L-shape, H-shape, bow-tie, spiral
- **Balance variation**: symmetric to highly asymmetric
- **All deterministic and 2D** (plottable)

## Visualizations

- Phase portraits: `results/claude_catalog_validation/plots/`
- Gallery: `results/claude_catalog_validation/gallery/system_gallery.png`
- Crossing comparison: `results/claude_catalog_validation/gallery/crossing_comparison.png`
