# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SKAE (Sparse Koopman Autoencoder) is a PyTorch research codebase for learning Koopman operator representations of nonlinear dynamical systems. The Koopman operator provides a linear representation of nonlinear dynamics in a lifted feature space, enabling linear prediction of nonlinear system evolution.

We are attempting to use a LISTA encoder instead of a MLP encoder for this Koopman autoencoder, which attempts to learn to make forecasts for nonlinear dynamical systems with multiple basins of attraction determined by having multiple fixed points. The idea is that the LISTA encoder should naturally enforce sparsity and allow for more simple and interpretable behavior.

We want to induce structured sparsity on the Koopman matrix such that each basin of attraction corresponds to its own high dimensional subspace where the Koopman dynamics are approximately linear. Our key insight is to enforce a sparse latent representation, structured as a union of subspaces, where each active support approximately corresponds to a specific basin of attraction or dynamical regime. Ideally, we would like to see if we can isolate distinct Koopman linear dynamics for each basin.

If this is true, we can isolate each basin and solve with LQR. Then, a nonlinear control problem over multiple basins of attraction reduces to solving linear Koopman dynamics within each basin using LQR, and modeling the changes between basins. We hypothesize that our periodic reencoding mechanism at inference time where we encode and immediately decode an input might be a good mechanism for modeling changes between basins.

More detailed notes about the project are in `docs/notes.tex`, which contains the working draft of the research paper that we will publish and contains any annotated notes. This essentially contains the current state of the project.

IMPORTANT NOTE: After any changes that we make to the framework or experiments that we run, we should ALWAYS
* make a descriptive git commit and push, and
* update `docs/notes.tex` by describing the experiment that was ran, and the results of the experiment. Also, interpret the results of the experiment in context if possible. When writing, carefully discern what has been implemented and explain it to me thoroughly. Check all the math. The tone of your writing should be academic and appropriate for an audience specializing in mathematics and machine learning. The writing should be for an award-winning paper at NeurIPS or ICML. Do not make any explicit references to code or filenames. Focus on explaining well in a scientific manner so that a reader can understand and reproduce the results.

## Documentation Updates
- Keep `docs/EXPERIMENTS.md` with a short **Current Status Summary** at the very top that states the problem(s) being solved, the solution if found (or the current approach if not), and clearly labels the **outstanding problem**.
- When new experiment results are produced, report and write them in this sequence:
  1. Report the concrete result(s).
  2. Explain the result(s) in the context of the experiment design/question.
  3. State the interpretation of the result(s).
  4. State implications for the overall project direction.
  5. Propose next steps.
- After reporting any results, update the project state in `docs/EXPERIMENTS.md` (including **Current Status Summary**, **Outstanding problems**, **Queue Status**, and the corresponding experiment entry).
- Note: In our intended **training/deployment** setting, we **do not** know the number of basins in advance or which trajectories belong to which basin. Avoid relying on ground-truth basin labels or fixed basin counts when proposing methods or interpreting results for training-time method design.
- For **evaluation on benchmark systems**, it is acceptable to use known basin counts and basin labels to measure separability/performance.
- Terminology/goal: prioritize **basin-support alignment** (each basin maps to a unique sparse support in latent `z`). Do not treat basin-block alignment as the primary objective unless explicitly required by a specific experiment.

## Directory Structure

```
skae/
├── skae/                  # Core library package
│   ├── __init__.py
│   ├── config.py          # Dataclass-based configuration system
│   ├── model.py           # Koopman machine implementations
│   ├── data.py            # Dynamical systems environments
│   ├── evaluation.py      # Model evaluation utilities
│   └── benchmarks/        # Benchmark system catalogs and adapters
├── tools/                 # CLI tools and scripts
├── scripts/               # Shell scripts for experiments (sbatch, etc.)
├── experiments/           # Experiment-specific code
├── tests/                 # Test suite
├── notebooks/             # Jupyter notebooks
├── docs/                  # Documentation
│   ├── notes.tex          # Research paper draft
│   ├── figures/           # Figures and visualizations
│   └── planning/          # Planning documents
└── runs/                  # Training outputs (gitignored)
```

## Common Commands

```bash
# Install dependencies (uses uv for package management)
uv sync
# Add a package
uv add <package_name>
```

## Architecture

### Core Components (in `skae/`)

- **`config.py`**: Dataclass-based configuration system with presets (`generic`, `generic_sparse`, `lista`, `lista_nonlinear`, `hyperlista`)
- **`model.py`**: Koopman machine implementations inheriting from `KoopmanMachine` base class
- **`data.py`**: Dynamical systems environments (Duffing, Pendulum, Lorenz63, Lyapunov, etc.) with `Env` base class
- **`evaluation.py`**: Comprehensive model evaluation with rollout strategies

### Key Abstractions

- **Encoder**: Maps observations x → latent z (sparse for LISTA variants)
- **Decoder**: Maps latent z → reconstruction x̂ (dictionary-based for LISTA)
- **Koopman Matrix K**: Linear dynamics in latent space: z_{t+1} = K @ z_t

## Coding Style & Naming Conventions

- Python, 4-space indentation, and PEP 8 style conventions.
- Use `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep module names short and descriptive (e.g., `evaluation.py`, `data.py`).
- Prefer small, composable helpers and keep CLI argument names consistent with existing tools.
- Configuration presets live in `skae/config.py`; prefer adding new presets over hard-coded arguments.
- Training artifacts are written to `runs/` and are not versioned; capture important results in `docs/` or `notebooks/`.

## Testing Guidelines

- Framework: `pytest`.
- Naming: tests live in `tests/` and follow `test_*.py` / `test_*` function names.
- Add tests alongside new model features or configuration options, especially for expected errors and shape checks.

## Commit & Pull Request Guidelines

- Commit messages follow Conventional Commits as seen in history: `feat: ...`, `docs: ...`, `fix: ...`.
- Keep the summary concise and in the imperative mood.
- PRs should include a short summary, key config changes (if any), and links to relevant runs or results (e.g., `runs/<model>/<timestamp>` or plots).
