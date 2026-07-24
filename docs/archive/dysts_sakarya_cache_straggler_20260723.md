# Sakarya cache straggler and guarded restart — 2026-07-23

## Concrete result

Dysts cache array job `10173636` completed 36 of 40 system/split elements.
The remaining four elements, `10173636_24` through `10173636_27`, were the
Sakarya train, validation, policy, and test caches. Each was canceled after
about 14 hours because its latest completed-trajectory count had not changed
for 10--13 hours. No partial Sakarya cache artifact had been published.
Dependency-held training job `10173637` and evaluation queue `10173638` were
canceled before allocation, so this failure consumed no GPU time.

## Experimental context

Each Sakarya split requires 200 independent trajectories with 30,000 retained
steps and 2,000 warm-up steps at the corrected sampling interval of 30 times
the native Dysts interval. Native Dysts integration used SciPy `solve_ivp`
with Radau and relative and absolute tolerances of \(10^{-12}\). The four
tasks requested four CPUs each but configured only two cache workers, so they
used about two CPU cores and roughly 1.0--1.1 GiB resident memory apiece.

## Interpretation

The workers were CPU-active rather than deadlocked. Progress stopping at
different trajectory indices across disjoint splits is consistent with a few
perturbed Sakarya initial conditions causing pathological adaptive-solver
work. An unchanged restart could encounter the same deterministic initial
conditions and repeat the failure.

## Project implications

The 36 completed non-Sakarya caches remain valid and are preserved. The first
guarded design kept Radau as primary and proposed a wall-time fallback to
DOP853. An independent audit rejected that design for the paper protocol
because a wall-time decision can depend on machine load. The audit also found
that Dysts 0.96 supplies Radau a standardized analytic Jacobian as `J / std`;
for standardized coordinates, the corresponding transform should scale both
rows and columns. DOP853 does not consume this Jacobian.

The final restart therefore uses DOP853 uniformly for every Sakarya
trajectory, with no solver fallback and no initial-condition filtering. It
keeps the exact deterministic split initial conditions, output grid, warm-up,
and relative and absolute tolerances of \(10^{-12}\). The 32-trajectory
DOP853 smoke cache completed in 10.8 seconds with all four CPU workers active,
compared with 295.6 seconds for the all-Radau smoke cache.

On the same 32 initial conditions, the first 200 retained steps
(\(5.9823\) physical-time units) had a worst per-trajectory normalized maximum
difference of \(4.95\times 10^{-8}\) and normalized RMSE of
\(2.18\times 10^{-10}\) between DOP853 and Radau. Across the complete
5,001-step smoke ensembles, normalized differences in coordinate means,
standard deviations, and 1/50/99-percentiles were respectively below
\(3.1\times 10^{-4}\), \(2.2\times 10^{-3}\), and \(9.5\times 10^{-3}\).
Both caches were finite and had exact shape \(32\times5001\times3\).

## Next steps

The focused suite passed 58 tests, and both the Radau and uniform-DOP853
Sakarya smoke caches completed. Rebuild only the four missing Sakarya split
caches with uniform DOP853 and four workers per task, then relaunch the full
training and evaluation dependency chain against the completed 40-cache set.

The replacement chain was submitted as Sakarya cache array `10178766`, full
training array `10178767` (dependent on the cache), and evaluation queue
`10178768` (dependent on training). The cache is CPU-only. Training requests
one GPU per active packed task, with 12 concurrent fits per GPU and at most 24
active GPUs; the preceding matched smoke sustained 99% GPU utilization.
Evaluation remains CPU-only.
