# Global-K residual forecast V2 smoke-gate failure (2026-07-21)

## Concrete result

The independently audited V2 chain used authorized prediction-card SHA-256
`a89f06ea60804e9c04359ca42b057adec476ecf5e8b66f75bec3f2cb23ee2bd6`,
task-manifest SHA-256
`86a3dce2ce8fd6ca569aebcccb6812ac6c3ee206ec21ba8e2ccf2642305fb024`,
source-manifest SHA-256
`a4bfe0715a5f08226b616805789354e276fa6319e2f702e33c7b75ea43492d15`,
and queue SHA-256
`fdf9105bb41522dfd0f8c2632923d136c55ed5e4eeda088079267c254a515e87`.
Queue, preparation, and outcome-blind A100 smoke jobs 10165734, 10165740,
and 10165741 completed. Smoke-gate job 10165742 failed. Science array
10165743, science telemetry 10165744, and summary 10165745 were canceled by
dependency. No scientific task ran and no forecast outcome was inspected.

The synchronized forecast computation lasted 48.187 seconds with 46 retained
samples. Mean utilization was 96.85%, p10 was 100%, the minimum 30-second
rolling mean was 97.45%, and peak memory was 40,699 MiB on an 81,920 MiB
A100. All compute-window, cadence, trace-bracketing, smoke-schema, method-roster,
finiteness, routing, null, and outcome-firewall checks passed. The sole failure
was whole-allocation mean utilization: 69.890625% against the unchanged 70%
floor. The miss was 0.109375 percentage points.

The outcome-free data manifest, outcome-blind smoke shard, compute-window,
raw-telemetry, and GPU-assessment SHA-256 values are
`9e87f25b3e84415ffae5d87931329fda989e55d27a4b9091a226105f8278e6e2`,
`d33b7edb2886b163dd6fd9e8b88fd6c1bf5ce227c8019491f068b846103afa4a`,
`2d1c8dd4c9bee43f249f18c4303a25f8c7481478016bf6bf825ab61fd0bf2afc`,
`1044f8e4190c3a276bf1c67ae6e093deb3d00b76b04a3244ffe333ea7e966f98`,
and
`7ea2bcccb8a6fdafff486a588071fc6d8117636f1ede36d5eb1b28f68432d320`.
The queue log SHA-256 is
`96d01852a0a8230412f36fc84dfea74bef36678fc42214a654593fbdfe50c175`.

## Context

V2 repaired V1's strict-JSON telemetry serialization and increased each smoke
and evaluation corpus from 8,192 to 98,304 distinct prospective trajectories.
It retained every scientific seed, checkpoint, routing/null rule, horizon,
metric, inferential unit, threshold, and decision gate. The larger workload did
produce sustained, meaningful A100 forecasting work and stayed well below the
memory limit. The fixed-cost authenticated loading, checkpoint audit, route
fitting, and null construction before the marked compute window nevertheless
left the allocation-wide average just below its conservative floor.

## Interpretation

V2 is operationally invalid, not a positive or negative scientific result.
The utilization gate is conjunctive and was frozen before execution; excellent
compute-window utilization cannot override the failed allocation-wide check.
The dependency firewall correctly prevented science. Nothing from this smoke
supports forecasting, support routing, coordinate specificity, local laws,
invariant subspaces, or sparse superiority.

## Project implications

Do not lower or reinterpret the 70% gate, inspect the smoke's hidden forecast
values, reuse its artifact bytes, or combine V2 with any successor. Preserve
`/network/scratch/l/lia/skae/global_k_residual_forecast_v2_20260721` unchanged.
The raw trace is useful only for a final outcome-blind resource calculation:
98,304 trajectories require about 48.2 seconds and 40.7 GiB while active.

## Next step

If the already-frozen scientific question is completed, permit at most one
fresh-root V3 operational retry, independently audited before launch. Keep the
70% threshold and every scientific choice unchanged; increase only the number
of distinct prospective trajectories enough to give a robust allocation-wide
margin while retaining safe A100 memory. Report V3 regardless of its result and
do not start a separate positive-seeking invariant-subspace campaign.
