# Kuramoto Recovery Plan

Date: March 5, 2026

## Current readout

The current intrinsic-HD baseline used the shared Seq8 settings from
`scripts/sweep_high_dim_benchmarks_seq8.sh`:

- `sequence_length=8`
- `target_size=256`
- `num_steps=10000`
- `sparsity_coeff=0.0025`
- dense or block-diagonal LISTA at `lista_alpha=0.1`, `lista_num_loops=5`
- current environment defaults: `kuramoto=16`, ring coupling, identical frequencies

Kuramoto `H1000` best-periodic results from
`results/high_dim_benchmarks_plan_seq8_20260305/forecasting_summary.md`:

| model | seed 0 | seed 1 | seed 2 | median |
|---|---:|---:|---:|---:|
| `generic_sparse` | `199.86` | `188.24` | `258.87` | `199.86` |
| `lista_blockdiag` | `154.44` | `258.50` | `1.069e6` | `258.50` |
| `lista_dense` | `6.636e8` | `3.083e7` | `7.063e11` | `6.636e8` |

Additional signals from the saved run artifacts:

- `generic_sparse` finishes with `spectral_radius ~ 1.011-1.012`.
- `lista_blockdiag` finishes with `spectral_radius ~ 1.026-1.036`.
- `lista_blockdiag` sparsity is only `~0.60-0.62`, below the usual target band.
- `lista_blockdiag` is competitive on two seeds and catastrophically fails on one seed.
- dense LISTA is not a recovery candidate.

## Interpretation

What is solid:

- Kuramoto is not solved at the current shared baseline.
- `lista_blockdiag` is the only LISTA-family variant close enough to justify follow-up.
- The current baseline is not a fair final verdict on Kuramoto because it reused shared coefficients rather than a Kuramoto-specific tuning pass.

What is likely but still an inference:

- The first failure mode is tuning and stability, not architecture impossibility.
- A representation mismatch may also be contributing: evaluation uses plain squared L2 error on raw state coordinates, while Kuramoto states are phase variables with periodic structure.
- Because the current environment uses identical frequencies, the global phase mode is neutral. If tuning-only recovery stalls, a phase-centered or sinusoidal observation map is the right next intervention.

## Recovery strategy

### Stage K0: Cheap tuning recovery on the existing environment

Goal:
- Test whether the current failure is mostly a poor LISTA operating point rather than a Kuramoto-specific representation problem.

Run:
- `lista_blockdiag` only
- current `kuramoto` defaults
- `target_size=256`
- `sequence_length=8`
- `num_steps=20000`
- `lista_alpha=0.15`
- `sparsity_coeff in {0.0005, 0.0010, 0.0025}`
- `lista_num_loops in {1, 3, 5}`
- `k_block_size=16`
- `seed in {0, 1, 2}`

Anchor:
- rerun `generic_sparse` on the same environment for `num_steps=20000`, `seed in {0,1,2}`

Rationale:
- This is the cheapest recovery that still tests the main hypothesis.
- It transfers the better Duffing operating region (`alpha=0.15`, lower sparsity pressure, lower loop counts) before introducing new modeling complexity.

### Stage K1: Promote only if K0 shows a stable block-diagonal region

Promote the best two K0 arms to:

- `num_steps=50000`
- `k_block_size in {8, 16, 32}`
- `seed in {0, 1, 2}`

Keep:
- the same `generic_sparse` anchor at `50000` steps

Goal:
- determine whether the remaining gap is training budget or structural.

### Stage K2: Representation intervention only if K1 stalls

Only do this if K1 still leaves block-diagonal LISTA unstable or clearly above the generic anchor.

Candidate interventions:

- phase-centered Kuramoto observations
- sine/cosine Kuramoto observations

Reason to defer:
- these are more invasive than a tuning recovery, so they should only be tried after the cheaper experiment fails.

## Acceptance criteria

K0 success:
- at least one `lista_blockdiag` arm has all three seeds finite
- median Kuramoto `H1000` best-periodic is below the current `generic_sparse` baseline (`199.86`) or very close to it
- no seed shows the current catastrophic-failure pattern (`~1e6` or worse)

K1 success:
- promoted `lista_blockdiag` arm has median `H1000` best-periodic below `150`
- worst-seed `H1000` best-periodic stays below `1e4`
- the arm remains the second baseline beside `generic_sparse` for stricter `N=32/64` runs

Escalate to Stage K2 if:
- K0 finds no stable region, or
- K1 still shows heavy-tail behavior across seeds

## Practical repo changes

The repo should contain:

- a dedicated Kuramoto recovery sweep script
- a matching collector script
- an experiment-log entry pointing to this plan

Dense LISTA stays dropped from the Kuramoto recovery path unless an explicit ablation requires it.
