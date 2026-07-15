# Reusable library map

`skae/` contains code that is reusable beyond the current paper. Frozen paper
rosters, evidence builders, and experiment-specific trainers live in
`experiments/neurips_2026/`.

## Feature map

| Feature | Maintained location | Notes |
|---|---|---|
| Sparse Koopman autoencoders | `model.py` | Encoders, decoders, latent operators, model construction |
| Environments and trajectory sampling | `data.py` | Common environment interface and dataset generation |
| Analytic multibasin systems | `dynamics/analytic/` | Registry, equations, integration, and retained system definitions |
| External Dysts systems | `benchmarks/dysts_adapter.py` | Dysts interface; deterministic cache profiles are in `dysts_cache_profiles.py` |
| Configuration | `config.py` | Dataclasses, presets, and environment parsing |
| Rollout metrics and diagnostics | `evaluation.py` | Shared evaluation behavior used by general and paper CLIs |
| Support families | `support/routing.py` | Support signatures, family fitting, and runtime assignment |
| Routed local operators | `support/local_operator.py` | Reusable local affine transition module |
| Training | `training/runner.py` | General `skae-train` implementation |
| Checkpoint evaluation | `cli/evaluate.py` | General `skae-evaluate` implementation |
| Historical checkpoint loading | `checkpoint_compat.py` | Compatibility for retained artifacts |
| Observation timesteps | `benchmarks/timesteps.py` | Shared timestep lookup |
| Portable runtime storage | `runtime_paths.py` | Shared scratch-root precedence used by library and paper workflows |
| Support-routed dynamics | `support/` | Parameterized masking, family construction, and local-map mechanics; no paper thresholds |

`claude_catalog/` and paper-focused modules under `benchmarks/` are temporary
import-compatibility namespaces. New code should use `skae.dynamics.analytic`
and `experiments.neurips_2026` directly.

## Dependency direction

```text
experiments/neurips_2026  --->  skae
scripts                   --->  installed CLI entry points
docs evidence builders    --->  experiments/neurips_2026
skae maintained code       -/->  experiments/neurips_2026
```

The reusable implementation must not import the current paper package. The
small historical compatibility shims under `skae/benchmarks/` are the sole
temporary exception; they forward old imports without owning any logic.

Compatibility names are boundary adapters, not a second API. Maintained code
uses the descriptive names in `experiments.neurips_2026.controlled`; immutable
historical names remain only in shims and artifact provenance so old
checkpoints can still be replayed.

## Environment names

- `analytic:<system>` is the maintained prefix for registered analytic systems.
- `dysts:<system>` selects an external Dysts system.
- Unprefixed controlled environments such as `gated_local_linear` remain part
  of the common environment interface.
- `claude:<system>` remains accepted only so historical configs and artifact
  identifiers replay exactly.

## Scientific invariants

- Methods used at training/deployment time cannot require known basin counts or
  basin assignments. Benchmark labels are evaluation-only.
- Dense no-sparsity baselines use tanh. Dense ReLU must be named as an ablation.
- Spatialized reaction--diffusion lifts satisfy `d_z >= 4 * d_x`.
- Alignment and routed-local-operator experiments use separately frozen family
  thresholds (`0.50` and `0.40`, respectively); these settings belong in the
  paper package, not as universal library defaults.

## Tests by area

- Models and shapes: `tests/test_model.py`, `tests/test_hyperlista.py`
- Environments and systems: `tests/test_data.py`, `tests/test_env_interface.py`
- Evaluation behavior: `tests/test_evaluation.py`
- Frozen paper contracts: `tests/test_paper_protocol.py` and the paper workflow
  tests
- Repository boundaries and CLI dispatch: `tests/test_repository_architecture.py`

Run all Python and pytest commands through `uv run` inside a compute allocation,
as specified in the root `AGENTS.md`.
