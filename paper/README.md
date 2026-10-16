# Reproducing the paper

Run every command from the repository root after `uv sync --locked`.
`uv run python -m paper <command> --help` lists the options of each command.

The order of work is: [define the runs](#how-an-experiment-is-defined), [run them](#running-experiments),
[collect the evidence](#collecting-the-evidence) once every run has finished, then
[build the tables](#building-the-tables) once the evidence is complete. To get the tables without training,
build them from the bundled evidence (`python -m paper rebuild`).

## How an experiment is defined

A **run** is named by four values, `<benchmark> <system> <model> <seed>`, for example
`multibasin cal_square_4 lista 0`. A **task file** has one such line per run; `tasks` writes it and
`train --tasks FILE` trains its lines. The paper has 2,925 runs, and the possible values are:

| Value | Choices | Defined in |
|---|---|---|
| benchmark | `multibasin`, `dysts` | `BENCHMARKS`, [`config.py`](config.py) |
| system | 15 multibasin systems (`cal_square_4`, `gated_local_linear`, ...) and 10 Dysts flows (`Chua`, ...) | `MULTIBASIN_SYSTEMS`, `DYSTS_NAMES`, [`systems/__init__.py`](systems/__init__.py) |
| model | `lista`, `lista_bd`, `lista_sb`, `sparse_mlp`, `sparse_mlp_bd`, `dense_mlp`; multibasin ablations `sparse_mlp_l1_0`, `dense_mlp_relu`, `lista_w64`, `sparse_mlp_w64`, `dense_mlp_w64` | `MODELS`, `ABLATIONS`, [`config.py`](config.py) |
| seed | 0–14 (0–4 for the `_w64` ablations) | `SEEDS`, `WINDOW64_SEEDS`, [`config.py`](config.py) |

`uv run python -m paper tasks` prints every run; `--benchmark`, `--systems`, `--models` and `--seeds`
filter them, and an unknown name is rejected with the list of valid ones.

**Settings.** `paper_config(benchmark, system, model, seed)` in [`config.py`](config.py) returns the run's
`RunConfig`, which is saved as the run's `config.json`. It has three parts:

| Part | Class | Holds |
|---|---|---|
| `model` | `ModelConfig`, [`skae/config.py`](../skae/config.py) | encoder, code size, LISTA settings, structure of `K`, loss weights |
| `train` | `TrainConfig`, [`config.py`](config.py) | number of updates, batch size, window length `L`, learning rates (`K` has its own), weight decay, schedule |
| `eval` | `EvalConfig`, [`config.py`](config.py) | test horizons, candidate re-encoding periods |

Every field has a one-line comment where it is defined. The defaults are the multibasin LISTA run;
`_multibasin_model`, `_dysts_model` and `paper_config` set the other runs. Settings shared by all runs
(numbers of test and validation starts, the period-selection horizon, the divergence rule, the seed table)
are constants at the top of [`config.py`](config.py). To see the full settings of a run:

```bash
uv run python -m paper train --benchmark dysts --system Chua --model lista --seed 0 --print-config
```

**Other settings.** For an experiment outside the paper, change the fields of a configuration and train it:

```python
from paper.config import paper_config
from paper.training.train import train

cfg = paper_config("multibasin", "cal_square_4", "lista", seed=0)
cfg.model.sparsity_weight = 1e-2
cfg.train.steps = 50_000
train(cfg, "runs/my_experiment/lista_l1_1e-2")    # writes config.json, final.pt and metrics.json there
```

Such runs are not part of the paper's roster, so `collect`, `evidence` and `tables` ignore them; read their
`metrics.json` directly.

## Tips for your own experiments

- **Updates.** You may not need the 200,000 updates of the multibasin runs; try fewer (`cfg.train.steps`).
- **Training windows.** Longer training windows can forecast better; 64-state windows (`cfg.train.window = 63`
  predicted steps, the `_w64` ablations) worked well in practice.
- **Re-encoding.** A good period for re-encoding is half to one training window, e.g. every 32 or 63 steps with
  64-state windows (`cfg.eval.periods`, or `reencode_every` of `skae`'s `forecast`).

## Running experiments

**Dysts data.** Dysts runs read trajectory caches that must be built once (multibasin data are simulated from
each run's seed):

```bash
uv run python -m paper data --workers 8          # data/dysts/<System>_{train,val,test}.pt
```

**One run.**

```bash
uv run python -m paper train --benchmark multibasin --system gated_local_linear --model lista --seed 0
```

The run writes `runs/multibasin/gated_local_linear/lista/seed_0/`: `config.json`, `train_log.csv`, `last.pt`
(restart state, every 1,000 updates), `final.pt` (the reported weights) and `metrics.json` (held-out forecasts).
Rerunning the command resumes from `last.pt`, and skips a finished run. `--steps N` shortens a run for a test
(written to `seed_0_steps<N>/`).

**Many runs.** Write a task file, then train one line per job, e.g. as a cluster array job:

```bash
uv run python -m paper tasks --out tasks.txt                                # the 2,925 paper runs
uv run python -m paper train --tasks tasks.txt --index "$TASK_INDEX"        # TASK_INDEX = 0, ..., 2924
```

Without `--index`, `train --tasks` trains every line in turn.

**GPUs.** Each run trains in one process on one device: `--device cuda` (the default; it falls back to the
CPU with a warning), `--device cuda:N` for GPU `N`, or `--device cpu`. A run does not span several GPUs, and
the models are small, so one run leaves most of a GPU idle. To use a GPU fully, run several processes on it at
once: in the shell, `&` starts each command as a separate background process and `wait` waits for all of them.

```bash
# four task lines at once, all on GPU 0
for i in 0 1 2 3; do
  uv run python -m paper train --tasks tasks.txt --index $i --device cuda:0 &
done
wait

# a cluster job with one GPU training lines 4k, ..., 4k + 3 (k = TASK_INDEX = 0, ..., 731 for the 2,925 lines)
for j in 0 1 2 3; do
  uv run python -m paper train --tasks tasks.txt --index $((4 * TASK_INDEX + j)) &
done
wait

# a machine with four GPUs: one run on each
for g in 0 1 2 3; do
  uv run python -m paper train --tasks tasks.txt --index $g --device cuda:$g &
done
wait
```

With many processes on one GPU, NVIDIA MPS (`nvidia-cuda-mps-control -d`) runs them more efficiently.

**Compute.**

| Benchmark | Runs | Updates | Per run, one NVIDIA L40S | Per run, 4 CPU cores |
|---|---|---|---|---|
| Multibasin | 2,025 | 200,000 | 8.5–18 min (18–21 min with 64-state windows) | 34–57 min |
| Dysts | 900 | 12,000 | 1.1–1.5 min, plus 35–45 s of data loading and evaluation | 0.5–1.5 h |

At one run per GPU this is about 450 GPU-hours; the paper's runs shared GPUs and took about 90 GPU-hours, plus
200 CPU-hours in total. Disk: 2.2 GB of Dysts caches, about 2 MB per run (6 GB in all) and 9 MB of evidence.

## Collecting the evidence

Run this once **every run has finished**. `evidence` collects the runs' forecasts, fits the classical
baselines and runs every analysis, writing compact files to `evidence/` with a `SHA256SUMS`
([`evidence/README.md`](evidence/README.md) lists them):

```bash
uv run python -m paper evidence --workers 16
```

`--force` replaces a non-empty `evidence/`. To spread the classical baselines over several jobs first, run
`baselines` with `--systems` and `--seeds`; `evidence` then skips their finished results under `runs/baselines/`.

| Producer | Cost |
|---|---|
| `data` | 46 min for the 30 caches, serially |
| `baselines` | multibasin 13 CPU-hours (225 tasks); Dysts about 100 hours (150 tasks of 20–65 min on 2 CPU cores) |
| `analyze` | `reencoding` 6 s per run (675 runs); `error-growth` 1.3 s per run on an L40S (450 runs); the others under 1 CPU-hour each |

## Building the tables

Build the tables **only after all runs and the evidence are complete**: each table cell aggregates the 15
seeds of every system, so tables built from part of the runs are not the paper's.

```bash
uv run python -m paper tables --evidence evidence --out results/tables
```

`collect`, `analyze`, `evidence`, `tables` and `rebuild` report which of the paper's 2,925 runs are missing
and exit with status 1 (`tables` then writes none of the affected tables) unless `--allow-missing-runs` is
given, which marks the affected cells `(missing runs)`. Shortened runs (`train --steps`) count only with
`--include-short` and are listed in `<evidence>/shortened_runs.txt`.

To build the tables from the bundled evidence instead, without training:

```bash
uv run python -m paper rebuild    # checks paper/evidence/SHA256SUMS, writes results/tables/
```

The 14 tables reproduce the paper's tables exactly.

## A quick end-to-end check

Two 200-step multibasin runs through every step, written under `quick_check/` (4–7 minutes on 4 CPU cores; 12 tables
with cells marked `(missing runs)` or `--`):

```bash
uv run python -m paper tasks --systems cal_square_4 --models lista dense_mlp --seeds 0 --out quick_check/tasks.txt
uv run python -m paper train --tasks quick_check/tasks.txt --steps 200 --device cpu --runs quick_check/runs
uv run python -m paper evidence --runs quick_check/runs --evidence quick_check/evidence --include-short --allow-missing-runs --force
uv run python -m paper tables --evidence quick_check/evidence --out quick_check/tables --allow-missing-runs
```

## Which evidence feeds which table

Evidence files are in `paper/evidence/` (bundled) or `evidence/` (recomputed); the producer of each is in
[`evidence/README.md`](evidence/README.md), and each table's code in [`tables/`](tables/).

| Table | `results/tables/` | Evidence |
|---|---|---|
| 1 | `table1_forecasting.tex` | `forecasting_{multibasin,dysts}.csv`, `baselines_{multibasin,dysts}.csv`, `support_transfer.csv` |
| 2 | `table2_interventions.tex` | `interventions.csv.gz` |
| 6, sparsity mechanisms | `sparsity_mechanisms.tex` | `forecasting_sparsity_ablation.csv`, `sparsity_mechanisms.json` |
| 7, re-encoding ablation | `reencoding_ablation.tex` | `reencoding.csv` |
| 8, training windows | `window_ablation.tex` | `forecasting_window64.csv`, `forecasting_multibasin.csv` |
| 9, exact matches | `alignment_counts.tex` | `support_transfer.csv` |
| 10, normalized entropy | `normalized_entropy.tex` | `support_transfer.csv` |
| 11, threshold sensitivity | `threshold_sensitivity.tex` | `support_sensitivity.json.gz`, `support_transfer.csv` |
| 12–14, multibasin per system | `per_system_multibasin_h{100,500,1000}.tex` | `forecasting_multibasin.csv`, `baselines_multibasin.csv` |
| 15–17, Dysts per system | `per_system_dysts_h{100,2000,4000}.tex` | `forecasting_dysts.csv`, `baselines_dysts.csv` |

The other appendix tables are hand-written. Figures are not regenerated; their numbers come from the evidence:
Figure 2 from `forecasting_{multibasin,dysts}.csv`, Figure 3 from `error_growth.json`, Figure 4 from
`interventions.csv.gz` and Figure 5b from `support_transfer.csv` (Figures 1 and 5a and the appendix vector
fields are illustrations).

## Notes

- **Hardware.** Training is bitwise reproducible on the same CPU type and thread count, or the same GPU type;
  elsewhere the last bits change, which chaotic Dysts rollouts amplify, so retrained runs match the bundled
  evidence up to rounding. `error_growth.json` is bitwise reproducible on an NVIDIA L40S. RBF EDMD is
  ill-conditioned, so `baselines` pins the BLAS threads (1 on multibasin, 2 on Dysts).
- **Versions.** `uv sync --locked` installs the paper's versions: torch 2.9.1 (CUDA 12.8), NumPy 2.1.2,
  SciPy 1.15.3, numba 0.63.1 and dysts 0.96.
- **Sakarya.** Besides the norm bound in the paper, a Sakarya draw is also rejected when the solver needs more
  than 711,580 right-hand-side evaluations (`SAKARYA_RHS_LIMIT` in [`systems/dysts.py`](systems/dysts.py)).
  Rejections replaced 4 of the 600 Sakarya trajectories.
- **Dysts baseline tiles.** The baselines fit 64-state tiles with stride 63; the last tile of a trajectory
  overlaps the one before it, so 51 of its 30,000 transitions count twice
  ([`baselines/dysts.py`](baselines/dysts.py)).
