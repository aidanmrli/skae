# High-Dimensional Benchmark Brief

Date: March 5, 2026

## One-slide summary

We chose three intrinsic high-dimensional environments because they stress different parts of the sparse-Koopman hypothesis:

- `kuramoto`: phase-locked twisted states in a genuinely high-dimensional oscillator system
- `hopfield`: multiple memory-retrieval basins with a controllable basin count
- `competitive_lv`: multiple survivor-set equilibria where different active species sets naturally map to different sparse supports

Current Seq8 baseline result at `target_size=256`:

| system | `generic_sparse` | `lista_blockdiag` | `lista_dense` | readout |
|---|---:|---:|---:|---|
| `competitive_lv` | `0.0651` | `0.1192` | `0.1654` | solved positive control |
| `kuramoto` | `199.86` | `258.50` | `6.636e8` | partial success, robustness problem |
| `hopfield` | `5436.67` | `3.599e15` | `3.045e33` | current blocker |

Metric:
- seed-median `H1000` best-periodic forecasting MSE

## Why these three systems

`competitive_lv`:
- best positive control for basin-support alignment
- each attractor corresponds to a different surviving-species set
- if the method works anywhere intrinsically high-dimensional, it should work here first

`kuramoto`:
- tests whether the model can handle multi-basin phase-locking in a scalable oscillator system
- dimension is intrinsic and easy to scale (`N=16`, then `32/64`)
- basin labels come from winding number, so evaluation is clean

`hopfield`:
- tests whether the method can separate multiple fixed-point memory basins
- basin count and dimension are independently controllable
- it is the hardest current benchmark because long-horizon stability is failing for every model family, not only LISTA

## What the current results mean

Main takeaways:

- `competitive_lv` validates that the intrinsic-HD benchmark idea is not broken. All three models can stay accurate at `H1000`, so the environment itself is not the problem.
- `kuramoto` is useful because it already separates architectures. `generic_sparse` is best, `lista_blockdiag` is plausible but not robust, and dense LISTA is not viable.
- `hopfield` is the real blocker. Even `generic_sparse` is poor, so this is not just a LISTA-specific tuning issue.

The current ranking:

1. `generic_sparse` is still the strongest baseline across the full intrinsic-HD suite.
2. `lista_blockdiag` is the only LISTA-family variant worth carrying forward on intrinsic-HD follow-up.
3. dense LISTA should be dropped from the intrinsic-HD path unless there is a very specific ablation question.

## Recommended presentation story

Slide 1:
- Motivation: current multi-basin tests were mostly 2D-with-padding; we needed intrinsically high-dimensional systems

Slide 2:
- Why these three environments
- one line each on attractor type:
  - phase-locked states
  - memory fixed points
  - survivor-set equilibria

Slide 3:
- results table above
- headline:
  - positive control passes (`competitive_lv`)
  - Kuramoto is promising but not robust
  - Hopfield is the main blocker

Slide 4:
- next experiments
- tune `lista_blockdiag` specifically for Kuramoto
- patch HyperLISTA before rerunning adaptive-threshold experiments
- only then queue stricter intrinsic-HD sizes (`N=32/64`)

## Short talk track

We now have an intrinsic high-dimensional benchmark suite rather than padded 2D systems. The suite is useful because the three environments stress different aspects of the basin-support hypothesis. `competitive_lv` acts as the positive control and is already working, which tells us the benchmark design is sound. `kuramoto` is not solved but is recoverable enough to justify a targeted LISTA follow-up. `hopfield` is the real long-horizon stability blocker and should be treated as the hardest benchmark in the current phase.
