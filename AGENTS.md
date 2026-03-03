# Repository Guidelines

## Documentation Updates
- Keep `docs/EXPERIMENTS.md` with a short **Current Status Summary** at the very top that states the problem(s) being solved, the solution if found (or the current approach if not), and clearly labels the **outstanding problem**.
- When new experiment results are produced, document them in this order:
  1. Report the concrete result(s).
  2. Explain the result(s) in the context of the experiment.
  3. Explain how to interpret the result(s).
  4. Explain project implications.
  5. Suggest next steps.
- After reporting results, update project state in `docs/EXPERIMENTS.md` (at minimum: **Current Status Summary**, **Outstanding problems**, **Queue Status**, and the relevant experiment log entry).
- Note: In our intended **training/deployment** setting, we **do not** know the number of basins in advance or which trajectories belong to which basin. Avoid relying on ground-truth basin labels or fixed basin counts when proposing methods or interpreting results for training-time method design.
- For **evaluation on benchmark systems**, it is acceptable to use known basin counts and basin labels to measure separability/performance.
- Terminology/goal: prioritize **basin-support alignment** (each basin maps to a unique sparse support in latent `z`). Do not treat basin-block alignment as the primary objective unless explicitly required by a specific experiment.

## Project Structure & Module Organization
This repository is a PyTorch-based research codebase for sparse Koopman autoencoders. The layout is organized around a core package plus CLI tooling.

Common paths:
- `skae/`: core library (models, data environments, evaluation utilities, config presets).
- `tools/`: CLI scripts used for training and evaluation (e.g., `tools/train.py`).
- `tests/`: pytest suites (`tests/test_*.py`).
- `experiments/` and `scripts/`: experiment code and shell scripts (sweeps/sbatch).
- `docs/` and `notebooks/`: research notes and exploratory analysis.
- `runs/`: training outputs (gitignored).

## Build, Test, and Development Commands
Use `uv` for reproducible environments.
- Always run Python entry points via `uv run`; do not invoke `python` or `python3` directly.

- `uv sync`: install dependencies from `uv.lock`.
- `uv pip install -e .`: editable install (alternative without lockfile).
- `uv run python tools/train.py --config generic_sparse --env duffing --sequence_length 1 --num_steps 20000`: example training run.
- `uv run python tools/evaluate_checkpoints.py --run_dir runs/<model>/<timestamp> --system lyapunov`: evaluate checkpoints.
- `pytest`: run the full test suite.
- `pytest tests/test_model.py -v`: run a focused test module.
- `pytest --cov=skae --cov-report=html`: coverage report (optional).

## SLURM Submission Rules
- Launch SLURM job scripts with `sbatch` (for scripts with `#SBATCH` headers); do not run them directly with `bash`.
- Prefer queue/launcher scripts that call `sbatch` to submit dependency chains (sweep -> collect -> compare).
- If a script is intended for SLURM execution, submit it via `sbatch scripts/<name>.sh` to ensure cluster-side permissions and environment are applied.
- In Codex/sandboxed sessions, `sbatch` submission requires out-of-sandbox escalation, so always run `sbatch` command with escalated permissions.

## Coding Style & Naming Conventions
- Python, 4-space indentation, and PEP 8 style conventions.
- Use `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep module names short and descriptive (e.g., `evaluation.py`, `data.py`).
- Prefer small, composable helpers and keep CLI argument names consistent with existing tools.

## Testing Guidelines
- Framework: `pytest`.
- Naming: tests live in `tests/` and follow `test_*.py` / `test_*` function names.
- Add tests alongside new model features or configuration options, especially for expected errors and shape checks.

## Commit & Pull Request Guidelines
- Commit messages follow Conventional Commits as seen in history: `feat: ...`, `docs: ...`, `fix: ...`.
- Keep the summary concise and in the imperative mood.
- PRs should include a short summary, key config changes (if any), and links to relevant runs or results (e.g., `runs/<model>/<timestamp>` or plots).

## Configuration & Outputs
- Configuration presets live in `skae/config.py`; prefer adding new presets over hard-coded arguments.
- Training artifacts are written to `runs/` and are not versioned; capture important results in `docs/` or `notebooks/`.
