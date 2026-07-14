# Core package map

`skae/` contains reusable model and benchmark implementation. Paper-specific
commands belong in `tools/`, and cluster orchestration belongs in `scripts/`.
The current scientific contract is the manuscript, not this map.

## Main modules

- `model.py`: encoders, decoders, latent transition operators, sparse Koopman
  autoencoders, and model construction.
- `data.py`: trajectory environments, controlled dynamical systems, reset
  distributions, and dataset generation.
- `config.py`: configuration dataclasses, presets, and environment timestep
  resolution.
- `evaluation.py`: rollout evaluation and diagnostic plotting shared by the
  maintained CLIs.
- `checkpoint_compat.py`: compatibility loading for retained historical
  checkpoints.

## Benchmark modules

- `benchmarks/paper_protocol.py`: frozen controlled and Dysts paper roster,
  model rows, seeds, budgets, and architecture-specific overrides. New paper
  task builders should import this contract instead of duplicating constants.
- `benchmarks/timesteps.py`: canonical observation-timestep lookup.
- `benchmarks/controlled_alignment.py`: frozen evaluation-only support-family
  construction, native/proxy label construction, tie-inclusive high-center-
  margin scoring, and alignment metrics.
- `benchmarks/transition_rich_basin_partition_manifest.py`: the exact 15
  controlled systems and six reported KAE recipes, plus lookup metadata used
  by evaluators.
- `benchmarks/dysts_adapter.py` and `dysts_cache_profiles.py`: external Dysts
  integration and deterministic cache settings. The paper Dysts roster lives
  only in `benchmarks/paper_protocol.py`.
- `claude_catalog/paper_systems.py`: the 13 analytic catalog systems retained
  by the controlled paper roster. The registry/factory API remains stable; the
  screening candidates were removed. The directory name is historical
  provenance, while paper-facing text uses descriptive system names.

## Non-negotiable experiment semantics

- Training and deployment methods cannot require a known basin count, a basin
  label, or a trajectory-to-basin assignment. Benchmark labels are allowed for
  evaluation; structured-transition rows that use the known count solely to
  choose a block count must remain labeled as diagnostics.
- Dense no-sparsity baselines use tanh hidden activations. Dense ReLU is an
  ablation, not the baseline.
- Spatialized reaction--diffusion lifts must satisfy `d_z >= 4 * d_x`.
- The paper's current forecasting summaries use test-set oracle selection over
  periodic re-encoding cadences and omit nonfinite steps. Do not silently call
  these values validation-selected or strict full-horizon errors.
- Alignment and routing use different frozen family definitions: the
  transductive alignment slice uses Jaccard `0.50`, while the staged route uses
  `0.40`.

## Where to test changes

- Model shapes and construction: `tests/test_model.py`,
  `tests/test_hyperlista.py`.
- Environments and deterministic systems: `tests/test_data.py`,
  `tests/test_env_interface.py`, and the retained transition-rich tests.
- Rollout semantics: `tests/test_evaluation.py`.
- Paper task contracts: `tests/test_paper_protocol.py` and the task-builder
  tests.

Run Python and pytest only through `uv run` on a compute node. See the root
`AGENTS.md` for allocation and SLURM rules.
