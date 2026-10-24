# Instructions for coding agents

## Map

`skae/` is the model package (NumPy and PyTorch only; what `pip install .` ships). `paper/` reproduces the
paper's 2,925 runs, their evidence and 14 tables with `python -m paper <command>` from the repository root.
The code map is in [README.md](README.md), the commands in [paper/README.md](paper/README.md) and the evidence
files in [paper/evidence/README.md](paper/evidence/README.md).

## Setup and checks

```bash
uv sync --locked                       # about 7 GB (CUDA torch); --no-default-groups: the model only
uv run python -m paper rebuild         # checksums, then the 14 tables in results/tables
uv run python examples/custom_data.py  # the model alone, under a minute on a CPU
```

For an end-to-end check, run the quick check in [paper/README.md](paper/README.md#a-quick-end-to-end-check)
(4–7 minutes on 4 CPU cores), not full runs.

## Invariants

Check these before and after a change; the outputs of the paper's code must not change.

| Invariant | Check |
|---|---|
| `paper/evidence/` matches its `SHA256SUMS` | `rebuild` verifies it |
| The tables reproduce the paper | `rebuild` before and after the change; `results/tables/` must be byte-identical |
| The run configurations are unchanged | `train ... --print-config` (or `paper_config(...).to_dict()` for every run of `paper_runs()`) before and after |
| Training numerics are unchanged | a short run (`train --steps 20 --no-eval --device cpu`) before and after, on the same machine: every tensor of `final.pt` identical |
| Without support losses (`support=None`, the default), `training_loss` and `fit` compute what they did before | the short run above, and a short `fit` with its defaults before and after: every tensor identical |
| `skae` imports nothing from `paper`, SciPy or dysts | `grep -rnE "^\s*(from\|import) (paper\|scipy\|dysts)" skae/` prints nothing |
| Producers write deterministic bytes in the bundled schema | `evidence` run twice on the same runs writes the same `SHA256SUMS` |

- Training numerics depend on the order of random draws, the dtypes (float32 model and data, float64 analytic
  integration), the `torch.vmap` stepping in `paper/systems/multibasin.py` and the operation order of
  `training_loss` and `run_updates`; keep them.
- Producers write through `paper/io.py` (`\n` line ends, no gzip timestamp, atomic writes); keep the column
  names and row order of every evidence file.

## Conventions that a change can break

- Codes are row vectors: one step is `z @ K`, so the stored `K` and LISTA `S` are transposes of the paper's.
- Time axis: second to last in `skae` and training windows (`[B, L + 1, d]`); first in the rollouts of
  `paper/evaluate.py` and `paper/data.py` (`[horizon, B, d]`).
- Seeds: the seed table in `paper/config.py` lists every generator seed; never change or reuse an offset.
- Lazy imports: `paper/__main__.py`, `paper/systems/__init__.py`, `paper/analysis/__init__.py`,
  `paper/baselines/cli.py` and `skae/__init__.py` import torch, SciPy and dysts only inside functions, so
  `--help` and `tasks` load without torch. An analysis module's docstring is its `--help`.
- The roster: `paper_runs()` in `paper/config.py` feeds `tasks`, the missing-run checks and the table coverage,
  so adding to `MODELS`, `ABLATIONS`, `SEEDS` or a system table changes the paper's runs.

## Common changes

| Change | Files |
|---|---|
| A multibasin system | `paper/systems/analytic.py`; a `SystemSpec` in `paper/systems/__init__.py` |
| A support loss | `skae/losses.py`; its hook in `training_loss` (`skae/models.py`) |
| An encoder | `skae/models.py`, `skae/config.py` (defaults keep the paper models unchanged); for paper runs `paper/config.py` and the labels in `paper/tables/` |
| A paper ablation | `paper/config.py` (`ABLATIONS`, `seeds_for`), `paper/training/collect.py` |
| The evaluation | `paper/evaluate.py`, `paper/training/evaluation.py` |
| An analysis | `paper/analysis/<name>.py`, registered in `ANALYSES` (`paper/__main__.py`); a row in both paper READMEs |
| A table | `paper/tables/<name>.py`, added to `builders` in `paper/tables/__init__.py` |
| A command or file name | `paper/__main__.py` and the READMEs |

## What not to do

- Do not edit `paper/evidence/`, and do not change the data, initialization, random-draw order or updates of a
  paper run.
- Do not add runs to the paper's roster for an experiment outside the paper; edit a `paper_config(...)` result
  and call `paper.training.train.train(cfg, run_dir)`, or use `skae` directly.
- Do not add figure code, trained runs or checkpoints.

Ask before changing anything the invariants protect, and report the checks you ran and their results.
