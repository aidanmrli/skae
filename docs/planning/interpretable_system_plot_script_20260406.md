# Interpretable System Plot Script

Date: April 6, 2026

## Objective

Add one reproducible plotting utility that can generate separate figures for the
three interpretable transition-rich toy systems discussed in this branch:

- `multiwell_strong_transition`
- `gated_local_linear`
- `gated_transfer_linear`

The script should support plotting any subset of those systems or all of them,
and each system should be saved separately.

## Design Choice

The current worktree no longer contains the full transition-rich environment
implementation that had existed earlier in the conversation, so the plotting
utility is implemented as a standalone tool rather than as a thin wrapper
around the environment factory.

That keeps the deliverable scoped and robust:

- one script
- one test file
- one saved overview figure per requested system

The standalone implementation follows the mathematical definitions already
documented in the transition-rich planning notes.

## Implementation

Added:

- [plot_interpretable_systems.py](/home/mila/l/lia/skae/tools/plot_interpretable_systems.py)
- [test_plot_interpretable_systems.py](/home/mila/l/lia/skae/tests/test_plot_interpretable_systems.py)

The CLI supports:

- `--systems all`
- `--systems multiwell_strong_transition,gated_local_linear`
- `--output_dir ...`
- `--grid_points ...`
- `--trajectory_length ...`
- `--start_points_per_axis ...`
- `--formats png,svg,pdf`

Each requested system is saved separately as:

- `<system_key>_interpretable_overview.<format>`

## Figure Contents

Each saved figure contains two panels:

- phase portrait: streamplot plus deterministic rollout trajectories
- region map: discrete region labeling for the same system

The three systems use distinct guides:

- `multiwell_strong_transition`: basin cores plus central transition corridor
- `gated_local_linear`: basin discs and sector-driven gating
- `gated_transfer_linear`: source circles, core circles, and transfer lanes

## Verification

Tests were written first and failed before the tool existed.

Focused test run after implementation:

```bash
uv run python -m pytest tests/test_plot_interpretable_systems.py -q
```

Result:

- `4 passed`

The script was also executed on all three systems with PNG output:

```bash
uv run python tools/plot_interpretable_systems.py \
  --systems all \
  --formats png \
  --grid_points 49 \
  --trajectory_length 80 \
  --start_points_per_axis 6 \
  --output_dir docs/figures/interpretable_systems_20260406
```

Generated example outputs:

- [multiwell_strong_transition_interpretable_overview.png](/home/mila/l/lia/skae/docs/figures/interpretable_systems_20260406/multiwell_strong_transition_interpretable_overview.png)
- [gated_local_linear_interpretable_overview.png](/home/mila/l/lia/skae/docs/figures/interpretable_systems_20260406/gated_local_linear_interpretable_overview.png)
- [gated_transfer_linear_interpretable_overview.png](/home/mila/l/lia/skae/docs/figures/interpretable_systems_20260406/gated_transfer_linear_interpretable_overview.png)

## Interpretation

This utility is good enough for rapid paper-branch visualization and for
recovering interpretable mechanics figures while the full transition-rich
environment implementation is not present in the worktree.

If later we restore the full transition-rich environment code, the next cleanup
step should be to decide whether this tool should remain standalone or be
rewired to use the canonical environment factory.
