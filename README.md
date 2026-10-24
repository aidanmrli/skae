# Sparse Koopman Autoencoders

Code for the NeurIPS 2026 paper
[*Sparse Koopman Autoencoders Identify Local Dynamical Regimes in Multibasin Systems*](https://arxiv.org/abs/2608.29057).

The repository has two parts:
- `skae/` which provides the model as a small PyTorch package
- `paper/` which provides code that reproduces our paper's runs, evidence and tables (`python -m paper <command>`).

## Install

```bash
git clone https://github.com/aidanmrli/skae.git && cd skae
uv sync --locked    # the model and the paper's dependencies (about 7 GB, mostly CUDA PyTorch)
```

For the model alone (NumPy and PyTorch): `pip install git+https://github.com/aidanmrli/skae.git`.
Python 3.10–3.13.

## Use SKAE on your data

```python
import torch
from skae import KoopmanAutoencoder, ModelConfig, fit, trajectory_windows

trajectories = ...  # float tensor [N, T, d]: N trajectories of T >= 64 equally spaced states
train, test = trajectories[:-8], trajectories[-8:]

torch.manual_seed(0)
model = KoopmanAutoencoder(ModelConfig(input_normalization=True), obs_dim=train.shape[-1])
model.set_input_normalization(train.mean(dim=(0, 1)), train.std(dim=(0, 1)).clamp_min(1e-8))

windows = trajectory_windows(train, window_length=63)     # [W, 64, d]: a state and the 63 after it
losses = fit(model, windows, epochs=100, batch_size=256)  # mean loss per epoch

model.eval()
with torch.no_grad():
    forecast = model.forecast(test[:, 0], horizon=50, reencode_every=32)  # [8, 50, d], predicts test[:, 1:51]
    support = model.encode(test[:, 0]).abs() > 1e-3                       # active code coordinates, [8, 256]
```

`ModelConfig()` is the paper's multibasin LISTA model; its docstring lists the settings of the
other models. `fit` is a plain Adam loop; optionally, `koopman_lr` gives `K` its own learning rate (the paper's
runs trained `K` 5 to 10 times more slowly) and `lr_schedule="cosine_tail"` decays the learning rates over the
last quarter of the updates (as in the paper's Dysts runs). Tips: 64-state training windows (`window_length=63`)
worked well in practice; re-encode every half to one training window (`reencode_every`); and you may need fewer
updates than the paper's 200,000 per multibasin run. A runnable version: [`examples/custom_data.py`](examples/custom_data.py).

### Optional support losses

We have two additional regularization terms that act on the active code coordinates: the temporal term penalizes
support changes within a window; the pair term matches supports to observation similarity. These loss terms were not
included in our paper but can be included for nicer latents that specialize better to basins
([`skae/losses.py`](skae/losses.py)). In our experiments with LISTA encoders without an L1 penalty, they made the
supports larger, more distinct between basins and slightly more consistent within them, and lowered the error of
forecasts without re-encoding, but not of forecasts that re-encode periodically. With an L1 penalty on the codes
(`sparsity_weight`), or at larger weights, they raised forecast errors.

```python
from skae import SupportLossConfig

cfg = ModelConfig(input_normalization=True, lista_output="none", lista_threshold="nonnegative",
                  lista_learn_thresholds=True, sparsity_weight=0.0,  # LISTA without L1
                  reconstruction_weight=1.0, loss_time_reduction="mean")  # the loss of our experiments
model = KoopmanAutoencoder(cfg, obs_dim=train.shape[-1])                  # then set_input_normalization as above
losses = fit(model, windows, epochs=100, batch_size=256, koopman_lr=2e-4, lr_schedule="cosine_tail",
             support=SupportLossConfig(temporal_weight=0.25, pair_weight=0.19))
```

## Code map

```text
skae/
  config.py           ModelConfig
  models.py           KoopmanAutoencoder (encode, rollout_latent, decode, forecast), LISTAEncoder, MLP, training_loss
  training.py         trajectory_windows, fit
  losses.py           the optional support losses: SupportLossConfig, WindowSimilarity
paper/
  __main__.py         the command line (python -m paper --help)
  config.py           paper_config(benchmark, system, model, seed): settings of every run; paper_runs(); seed table
  systems/            the 15 multibasin systems (analytic.py, gated.py) and the 10 Dysts flows (dysts.py)
  data.py             multibasin training windows, hard initial conditions, validation and test starts
  training/           train.py (trainer, resume), evaluation.py (metrics.json), collect.py (runs -> evidence)
  evaluate.py         load_run_model; re-encoded rollouts, divergence fallback, forecast MSE, period selection
  baselines/          DMD, polynomial and RBF EDMD
  analysis/           support families and basin alignment (support.py, support_transfer.py), interventions,
                      re-encoding ablation, sparsity mechanisms, error growth, statistics (stats.py)
  tables/             one module per paper table or table family; common.py: IQM over seeds, means, marks
  evidence/           bundled results of the paper's runs (read by tables/)
AGENTS.md             conventions and checks for changing the code
```

## Reproduce the paper

All commands run from the repository root; `uv run python -m paper <command> --help` lists the options.

**The tables, from the bundled evidence** (under a minute on a CPU, no training):

```bash
uv run python -m paper rebuild    # writes the 14 tables to results/tables/
```

**A single run** (about 11 minutes on an NVIDIA L40S; add `--device cpu` to train on a CPU,
`--steps 200` for a quick test):

```bash
uv run python -m paper train --benchmark multibasin --system gated_local_linear --model lista --seed 0
# -> runs/multibasin/gated_local_linear/lista/seed_0/: config.json, final.pt, metrics.json
```

**A subset of experiments**, for example LISTA and Dense MLP on one system with three seeds (six runs):

```bash
uv run python -m paper tasks --benchmark multibasin --systems cal_square_4 --models lista dense_mlp --seeds 0 1 2 --out tasks.txt
uv run python -m paper train --tasks tasks.txt
uv run python -m paper evidence --allow-missing-runs    # runs/ -> evidence/
uv run python -m paper tables --evidence evidence --allow-missing-runs
```

Cells that lack some of the paper's runs are marked `(missing runs)`.

**All experiments** (2,925 runs; about 90 GPU-hours and 200 CPU-hours as run for the paper):
[`paper/README.md`](paper/README.md).

## Citation

Please consider citing our work if you found it helpful:

```bibtex
@inproceedings{
  li2026sparse,
  title={Sparse Koopman Autoencoders Identify Local Dynamical Regimes in Multibasin Systems},
  author={Aidan Li and Uday Kiran Reddy Tadipatri and Mahan Fathi and Sarath Chandar and Ross Goroshin},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=oenYjKFj4D}
}
```

## License

MIT license. [`LICENSE`](LICENSE).
