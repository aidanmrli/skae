# Benchmark Extension Coordination

Date: May 12, 2026

## Shared rule

Both new benchmarks must produce a working one-seed LISTA-family result before
any broad multi-system, multi-seed, or large-scale rollout generation. Labels
remain evaluation-only: basin labels, basin counts, outcome labels, and contact
phase labels must not be used for training, support-family construction, model
selection, or method design.

## Workstreams

- Spatialized multibasin reaction-diffusion fields:
  progress doc: `docs/SPATIALIZED_REACTION_DIFFUSION_BENCHMARK.md`
  first target: one source system, one seed, minimal grid/resolution if needed,
  LISTA-family model, forecast MSE plus support-family/final-basin read.
- ManiSkill insertion:
  progress doc: `docs/MANISKILL_INSERTION_BENCHMARK.md`
  first target: state-only `PegInsertionSide-v1`, one seed or one small
  rollout packet, controlled LISTA-family model or closest supported path,
  rollout metrics plus support/outcome or support/contact read.

## Run policy

- Use `uv run` for Python entry points.
- Do not run Python, pytest, data collection, training, or evaluation on the
  login node.
- Use `salloc` for interactive compute-node checks or `sbatch` for SLURM
  scripts.
- Do not launch large arrays until the corresponding one-seed smoke path works.

## Current status

- Separate worker agents have been assigned to the PDE and ManiSkill
  workstreams.
- Parent coordination is limited to integration, documentation alignment, and
  checking that each worker preserves the one-seed-first rule.
- One-seed jobs have completed. Spatialized reaction-diffusion smoke job
  `9530539` completed with finite but non-publishable metrics. ManiSkill
  prepare job `9530626` and controlled LISTA train/eval job `9530627` both
  completed; rollout metrics are finite, but support discovery collapsed to
  one family.
- One-seed tuning improved both workstreams. PDE tuning fixed support collapse
  and reached `H=12` final-basin consistency `0.917` with compressed
  support-family NMI `0.709` at Jaccard `0.7` on all held-out test
  trajectories; final majority fraction is now treated as diagnostic
  stratification rather than a required filter. ManiSkill tuning fixed the one-family collapse
  and reached outcome NMI `0.347` at the best overfragmented read, with a more
  compact `131`-family read at outcome NMI `0.303`.
- Same-seed dense controls completed as jobs `9553549` (spatialized PDE) and
  `9553550` (ManiSkill), with support-threshold sweeps `9553603` and
  `9553602`. PDE is mixed: dense forecasts better, but LISTA has cleaner
  compact basin-support alignment. ManiSkill is more promising: LISTA is much
  better at H100 rollout and keeps a compact outcome/contact support signal,
  while dense needs heavy overfragmentation to approach outcome NMI.
- Current decision: neither benchmark should be promoted as main paper
  evidence yet. PDE needs a convolutional model-family comparison; ManiSkill
  needs perturbation-balanced state rollouts before any RGB-D expansion.
- The first ManiSkill prepare submission `9530510` failed immediately because
  `mani_skill` was absent from the active `uv` environment; the prepare runner
  now uses `uv run --with mani_skill --with h5py` for download/replay/compact
  data construction without adding ManiSkill to the base project dependency
  set.
- ManiSkill simulator replay is currently opt-in because replay on the CPU
  node failed at SAPIEN/Vulkan initialization; the successful prepare path
  compacts raw downloaded `env_states`.
