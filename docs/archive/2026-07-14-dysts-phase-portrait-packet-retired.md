# Retired Dysts phase-portrait packet

On 2026-07-14, the directory
`docs/figures/dysts_dt30_phase_portraits_seed0_h1000_h5000_all_models_20260501/`
was removed from the active tree during the repository-architecture cleanup.

The packet contained 24 rendered PDF/PNG files (about 111 MB) covering 12
source systems, six model variants, seed 0, and horizons 1000--5000. It was not
included by `docs/neurips_sparse_koopman_multibasin.tex`; two systems were
outside the retained 10-system Dysts roster; and its manifest referenced
historical source CSV/rollout files that were no longer versioned. It therefore
could not serve as a self-contained reproducible paper artifact.

The current Dysts claims use compact rows and generated displays under
`docs/figures/neurips_paper_2026/`. The retired packet remains recoverable from
Git commit `e1df75d9a47883fa22e377d03c8e194447ef96bf`.
