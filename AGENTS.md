# Repository Guidelines

## Documentation Management
- Organize active documentation around `docs/neurips_sparse_koopman_multibasin.tex`, which is the paper-facing source of truth for claims, narrative, experiment ordering, display planning, and interpretation.
- Use `docs/README.md` as the compact map of active documentation responsibilities. Do not create parallel status trackers with overlapping ownership.
- Active authored text files outside `docs/archive/` must stay at or below 500 lines. If a file would exceed that cap, either split it by a single clear purpose or move historical detail to `docs/archive/`.
- Keep each active document responsible for one purpose:
  - `docs/neurips_sparse_koopman_multibasin.tex`: main paper draft and current paper claims.
  - `docs/appendix/*.tex`: single-purpose appendix fragments included by the draft.
  - `docs/figures/neurips_paper_2026/`: active paper figures, tables, and generated display artifacts.
  - `docs/archive/`: historical plans, logs, handoff notes, old drafts, old literature notes, build artifacts, and superseded status trackers.
- Do not resurrect `docs/EXPERIMENTS.md`, `docs/PAPER_TRACK_STATUS.md`, or `docs/PAPER_EXPERIMENT_EVIDENCE_MAP.md` as active sources of truth. If old material is needed, consult the archived copy and migrate only the paper-relevant claim or protocol detail into the draft or the appropriate appendix fragment.
- When new experiment results are produced, document them in this order:
  1. Report the concrete result(s).
  2. Explain the result(s) in the context of the experiment.
  3. Explain how to interpret the result(s).
  4. Explain project implications.
  5. Suggest next steps.
- After reporting results, update the active paper source if the result affects paper claims, narrative, display planning, protocol, or priorities. If the result is useful provenance but not part of the current paper, add a dated archive note instead of expanding an active tracker.
- Project phase: we are actively trying to wrap up this project and convert the evidence into a publishable top-tier machine learning conference paper, with NeurIPS as the default target venue. Prioritize work that directly strengthens that paper.
- Current research focus: Handing over the documents to senior coauthors with emphasis on clear descriptions of all experimental protocol that isolates causal effects and does not include code names. The aim is to make the choices and current experimental results clear to senior coauthors so they can decide how to position the paper.
- Note: In our intended **training/deployment** setting, we **do not** know the number of basins in advance or which trajectories belong to which basin. Avoid relying on ground-truth basin labels or fixed basin counts when proposing methods or interpreting results for training-time method design.
- For **evaluation on benchmark systems**, it is acceptable to use known basin counts and basin labels to measure separability/performance.
- Terminology/goal: prioritize **basin-support alignment** (each basin maps to a unique sparse support in latent `z`). Do not treat basin-block alignment as the primary objective unless explicitly required by a specific experiment.
- Dense baseline activation rule: dense no-sparsity baselines must use tanh
  hidden activations by default. Do not launch or document a dense ReLU model
  as the dense baseline; if a ReLU dense model is intentionally tested, label
  it explicitly as a ReLU ablation, not as the baseline.
- Spatialized multibasin PDE rule: Koopman lifting experiments must use an overcomplete latent. For the spatialized reaction-diffusion benchmark, enforce `d_z >= 4 * d_x`, where `d_x = channels * grid_size^2` (`channels=2` for the current PDE fields). Examples: grid `16` requires `d_x=512` and `d_z>=2048`; grid `32` requires `d_x=2048` and `d_z>=8192`. Do not launch or document new spatialized PDE runs with an undercomplete latent.

## Project Structure & Module Organization
This repository is a PyTorch-based research codebase for sparse Koopman autoencoders. The layout is organized around a core package plus CLI tooling.

Common paths:
- `skae/`: core library (models, data environments, evaluation utilities, config presets).
- `tools/`: CLI scripts used for training and evaluation (e.g., `tools/train.py`).
- `tests/`: pytest suites (`tests/test_*.py`).
- `experiments/` and `scripts/`: experiment code and shell scripts (sweeps/sbatch).
- `docs/`: active NeurIPS draft sources plus archived historical notes.
- `notebooks/`: exploratory analysis.
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

## Compute Node Policy
- **NEVER run programs directly on the login node.** This includes Python scripts, training runs, evaluation scripts, validation sweeps, and any compute-intensive work.
- Every program execution must go through a compute node. Use `salloc` to get an interactive allocation before running:
  - CPU-only: `salloc --mem=8G -c 4 --partition=long`
  - With GPU: `salloc --gpus 1 --mem=8G -c 4 --partition=long`
- After `salloc` grants a node, run your command inside that allocation.
- This applies to all `uv run python ...` invocations, `pytest`, and any other non-trivial process. The exceptions are lightweight git, file editing, shell commands, and manuscript/PDF builds such as `latexmk`, `pdflatex`, and `bibtex`.

## SLURM Submission Rules
- Launch SLURM job scripts with `sbatch` (for scripts with `#SBATCH` headers); do not run them directly with `bash`.
- Use the `long` partition by default for SLURM jobs. Treat `long` as the repository convention unless a specific script or experiment has a documented reason to use another partition.
- Prefer queue/launcher scripts that call `sbatch` to submit dependency chains (sweep -> collect -> compare).
- Before launching any SLURM job that requests GPUs, read the job script and the called code to verify expected GPU utilization is high; identify and fix any likely GPU wastage (unnecessary GPU requests, oversized GPU counts, CPU-bound stages, undersized batching, reduce or eliminate idle GPU waits, missing CUDA device use, or tiny workloads), and call a subagent for this audit when useful with the goal of maximizing GPU utilization.
- If a script is intended for SLURM execution, submit it via `sbatch scripts/<name>.sh` to ensure cluster-side permissions and environment are applied.
- Do not override a script away from its existing `long` defaults without checking that the alternative partition is actually required and will not impose a stricter GPU/QOS limit.
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
