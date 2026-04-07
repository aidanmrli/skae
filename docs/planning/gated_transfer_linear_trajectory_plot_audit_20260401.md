# Gated Transfer Linear Trajectory Plot Audit

Date: April 1, 2026

## Objective

Check how the existing trajectory figures for `gated_transfer_linear` were made,
explain why they look confusing, and generate clearer ground-truth trajectory
views.

## Concrete results

- The existing figure script is
  [plot_chart_switching_transfer_system.py](/home/mila/l/lia/skae/tools/plot_chart_switching_transfer_system.py).
- The current trajectory figures were generated from a deterministic uniform
  `10x10` grid over
  `[-0.9 * init_range, 0.9 * init_range]^2 = [-2.52, 2.52]^2`, not from an
  irregular sample.
- The rollout length in the figure packet is `90` steps, using the true
  environment `env.step` and
  [generate_trajectory()](/home/mila/l/lia/skae/skae/data.py).
- The trajectory figures plot all `100` starts, but the transfer-summary panel
  only counts starts inside source neighborhoods.
- On the plotted `10x10` grid, only `8/100` starts lie inside source
  neighborhoods:
  - source `0`: `2`
  - source `1`: `3`
  - source `2`: `3`
- On the calibration `17x17` grid, the source-neighborhood counts are the
  intended `9/9/9`.
- A denser source-neighborhood visualization now exists with `144` starts
  total:
  - source `0`: `47`
  - source `1`: `48`
  - source `2`: `49`

## Result In Experimental Context

The original figure packet mixes three different notions:

- all-start qualitative trajectory plots from the coarse `10x10` grid
- source-neighborhood transfer counts computed only from the subset of starts
  inside the dashed source circles
- calibration metrics reported from a separate `17x17` grid

That makes the original figures easy to misread.

## Interpretation

- The trajectories were sampled from a uniform grid.
- The reason they do not look like a clean uniform spread of transfers is that
  most of the `10x10` starts are outside the source neighborhoods and are
  therefore irrelevant to the source-to-endpoint transfer summary.
- The jagged or kinked appearance is not primarily a plotting bug. It comes
  from the system definition: `gated_transfer_linear` is a hard piecewise-
  affine system with discrete chart switching between `core`, `return`, `exit`,
  and `channel` regions, so trajectories can change direction abruptly at
  region boundaries.
- The chart-colored figure is especially visually busy because each two-point
  segment is recolored by the active chart.

## Project Implications

- The old figure packet is mechanically faithful but not ideal for
  communication.
- For paper-facing use, the trajectory plots should make the sampling protocol
  explicit and should separate:
  - all-start qualitative ground-truth trajectories
  - source-neighborhood-only starts that actually define the transfer metric
- If we want trajectories that look smoother, that is a system-design or
  integration-choice issue, not just a figure-style issue.

## New Figure Artifacts

The following clearer ground-truth figures were added under
[chart_switching_transfer_20260331](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331):

- [gated_transfer_linear_uniform_start_grid_10x10.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_uniform_start_grid_10x10.svg)
  - shows the actual uniform `10x10` start grid used by the original plotting
    script
  - highlights which starts are inside source neighborhoods
- [gated_transfer_linear_ground_truth_trajectories_10x10_starts.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_ground_truth_trajectories_10x10_starts.svg)
  - shows true simulator trajectories from that same uniform `10x10` grid
  - marks every start point explicitly
- [gated_transfer_linear_ground_truth_source_starts_17x17.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_ground_truth_source_starts_17x17.svg)
  - shows only source-neighborhood starts from the calibration `17x17` grid
  - this is the clearest view of the transfer geometry used by the benchmark
    summary
- [gated_transfer_linear_dense_source_trajectories.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_dense_source_trajectories.svg)
  - shows many more starts inside the dashed source circles
  - makes it clear that the system itself funnels many shell starts into narrow
    transfer channels before settling
- [gated_transfer_linear_flow_field.svg](/home/mila/l/lia/skae/docs/figures/chart_switching_transfer_20260331/gated_transfer_linear_flow_field.svg)
  - shows the ground-truth directional vector field on a regular grid over the
    full state plane
  - confirms that the line-like transport between circles is a property of the
    system definition, not a trajectory-plotting bug

## Reproducibility Note

The audit figures were generated directly from the ground-truth simulator with
`uv run python`, using:

- `env = make_env(Config with ENV_NAME='gated_transfer_linear')`
- uniform `10x10` and `17x17` deterministic grids
- `generate_trajectory(env.step, initial_state, length=90)` for the qualitative
  `10x10` plot
- `generate_trajectory(env.step, initial_state, length=180)` for the
  source-neighborhood `17x17` plot
- `generate_trajectory(env.step, initial_state, length=180)` for the denser
  `9x9` per-source bounding-grid plot
- `(env.step(x) - x) / dt` on a `21x21` full-state grid for the flow field
