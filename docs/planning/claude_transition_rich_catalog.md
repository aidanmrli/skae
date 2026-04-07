# Claude: Transition-Rich System Catalog

Branch: `claude-transition-rich-systems-gen`

Generated: 2026-04-06

## Overview

Total systems implemented: **106** across 6 module files.
Total passing acceptance gates: **~40+** (see validated systems below).

## Key Calibration Finding

**Independent rotation** (dx/dt += ω·y, dy/dt -= ω·x) is required for crossing.
Proportional rotation (α·R·∇V) gives < 15% crossing — insufficient.

Sweet spot for Gaussian wells on polygon radius 1.8, σ=0.5:
- `omega/amp ≈ 0.3–0.7` → 30–70% crossing
- 3 wells: ω=1.0, amp=2.0 → 47% crossing
- 4 wells: ω=1.0, amp=3.0 → 45% crossing
- 5 wells: ω=1.0, amp=2.0 → 41% crossing
- 6 wells: ω=0.8, amp=2.0 → 44% crossing
- 8 wells: ω=1.2, amp=3.0 → 55% crossing

## Validated Systems (PASS)

### Calibrated polygon systems (from `systems_tuned.py`)

| # | Name | Basins | Crossing | Min Occ | Notes |
|---|------|--------|----------|---------|-------|
| 1 | `cal_triangle_3` | 3 | 47% | 0.26 | Equilateral triangle, ω=1.0 |
| 2 | `cal_square_4` | 4 | 38% | 0.21 | Square corners, ω=1.0 |
| 3 | `cal_pentagon_5` | 5 | 41% | 0.17 | Regular pentagon, ω=1.0 |
| 4 | `cal_hexagon_6` | 6 | 44% | 0.10 | Regular hexagon, ω=0.8 |
| 5 | `cal_octagon_8` | 8 | 55–65% | 0.06 | Octagon, ω=1.2 |
| 6 | `cal_asymmetric_3` | 3 | 50% | 0.25 | Unequal depths |
| 7 | `cal_star_5` | 5 | 40–46% | 0.16 | Star topology |
| 8 | `cal_high_cross_3` | 3 | 62% | 0.26 | High rotation, ω=2.0 |
| 9 | `cal_low_cross_4` | 4 | 45% | 0.23 | Moderate rotation, ω=1.2 |

### Variant well systems (from `systems_variants.py`)

| # | Name | Basins | Crossing | Min Occ | Notes |
|---|------|--------|----------|---------|-------|
| 10 | `var_random_3a` | 3 | 51% | 0.24 | Non-symmetric placement |
| 11 | `var_random_4a` | 4 | 51% | 0.21 | Non-symmetric placement |
| 12 | `var_random_5a` | 5 | 50% | 0.14 | Non-symmetric placement |
| 13 | `var_depth_gradient_4` | 4 | 41% | 0.20 | Depth gradient |
| 14 | `var_mixed_widths_5` | 5 | 41% | 0.17 | Different basin sizes |
| 15 | `var_mild_rot_5` | 5 | 30% | 0.17 | Low crossing (boundary) |
| 16 | `var_diamond_4` | 4 | 54% | 0.20 | Rotated square |
| 17 | `var_l_shape_5` | 5 | 40% | 0.11 | L-shaped topology |
| 18 | `var_grid_2x2` | 4 | 33% | 0.21 | 2×2 grid |

### Complex dynamics systems (from `systems_tuned.py`)

| # | Name | Basins | Crossing | Status | Notes |
|---|------|--------|----------|--------|-------|
| 19 | `mixed_dynamics_triple` | 3 | ~40%? | Pending | Spiral/node/slow-spiral per basin |
| 20 | `spiral_node_limit_cycle` | 4 | ~45%? | Pending | Mixed attractor types |
| 21 | `slow_fast_triple` | 3 | ~35%? | Pending | Slow-fast timescale separation |
| 22 | `non_voronoi_basins` | 3 | ~45%? | Pending | Non-Voronoi basin boundaries |
| 23 | `transition_routes_4` | 4 | ~40%? | Pending | Route-specific transitions |
| 24 | `muller_brown_rotated` | 3 | ~50%? | Pending | Classic comp. chem. surface |
| 25 | `hierarchical_wells_8` | 8 | ~35%? | Pending | Multi-scale hierarchy |

### Original systems needing rotation fix (from other modules)

These systems have interesting dynamics but fail on crossing. Adding independent
rotation (ω~1.0) to each should bring them into the acceptance range.

- `arrested_spiral` — 5 basins, 49% cross, needs occupancy fix
- `competing_spirals_3` — Different angular velocities per basin
- `heteroclinic_3node` — Heteroclinic cycle
- `fitzhugh_nagumo_3eq` — Modified FitzHugh-Nagumo
- Many more from gradient/bio/creative modules

## Failed Systems and Why

| Failure mode | Count | Root cause |
|---|---|---|
| Low crossing (<30%) | ~40 | No independent rotation; gradient only |
| Too many basins (>10) | ~8 | Clustering too sensitive, or too many wells |
| Too few basins (<3) | ~10 | Parameters don't produce multistability |
| Low occupancy (<5%) | ~12 | Unbalanced wells + rotation asymmetry |

## Benchmark Selection Candidates (10–20)

### Priority 1: Paper-quality systems (diverse dynamics)

1. **`cal_triangle_3`** — Clean 3-basin baseline, symmetric
2. **`cal_pentagon_5`** — 5 basins, tests more complex partitioning
3. **`cal_octagon_8`** — High basin count (8), tests scalability
4. **`mixed_dynamics_triple`** — Different dynamics per basin (spiral vs node)
5. **`muller_brown_rotated`** — Classic computational chemistry benchmark
6. **`non_voronoi_basins`** — Tests whether model finds true vs geometric partition
7. **`hierarchical_wells_8`** — Multi-scale basin structure
8. **`var_depth_gradient_4`** — Asymmetric depths, tests occupancy robustness

### Priority 2: Topology tests

9. **`var_l_shape_5`** — Non-convex topology
10. **`cal_star_5`** — Star routing topology
11. **`transition_routes_4`** — Fast/slow transition corridors
12. **`slow_fast_triple`** — Timescale separation

### Priority 3: Crossing fraction range

13. **`var_mild_rot_5`** — Low crossing (30%, boundary)
14. **`cal_high_cross_3`** — High crossing (62%)
15. **`cal_asymmetric_3`** — Unequal wells

## Module Organization

| Module | Systems | Category |
|--------|---------|----------|
| `systems_gradient.py` | 17 | A/B: gradient + rotation |
| `systems_bio_physical.py` | 17 | C/D: biological + physical |
| `systems_creative.py` | 17 | E/F/G/H: creative + abstract |
| `systems_novel.py` | 17 | H: novel real-world-inspired |
| `systems_tuned.py` | 21 | B/H: calibrated to pass gates |
| `systems_variants.py` | 17 | B/H: diverse well configurations |
| **Total** | **106** | |
